"""OpenRouter provider implementation."""

import dataclasses
import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any
from urllib.parse import quote

import httpx

from utils.env import get_env, suppress_env_vars

from .openai_compatible import OpenAICompatibleProvider
from .openrouter_batch_wire import (
    SUBMISSION_OUTCOME_UNKNOWN,
    build_chat_body,
    parse_listing,
    parse_results,
    parse_status,
    raise_for_http_error,
    reject_invalid_temperature,
    reject_non_text,
    serialize_envelope,
    strip_batch_suffix,
)
from .registries.openrouter import OpenRouterModelRegistry
from .shared import (
    BatchListing,
    BatchOutcome,
    BatchProviderError,
    BatchRequest,
    BatchStatus,
    CancelOutcome,
    CancelState,
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
    validate_batch_requests,
)
from .shared.batch_catalog import (
    BatchCatalog,
    BatchFamilyRule,
    get_batch_catalog_source,
    unclassified_newer_listings,
    warn_once,
)
from .shared.batch_resolution import (
    PARAMETERS_FIELD,
    PUBLISHED_CAP_FIELD,
    BatchModelResolution,
    apply_listed_parameters,
    apply_published_bounds,
    resolve_capabilities,
)
from .shared.published_limits import apply_published_limits

#: Sentinel: "the caller did not capture a catalog, ask the source".
_CATALOG_UNSET: Any = object()


class BatchModelRefused(ValueError):
    """The batch adapter deliberately refuses a model (no batch listing, a variant, an empty family).

    Distinct from the plain ``ValueError`` of an unknown or restricted model so admission can explain
    *why* a model cannot be batched without misreporting a model the provider does not serve at all.
    """


class OpenRouterProvider(OpenAICompatibleProvider):
    """Client for OpenRouter's multi-model aggregation service.

    Role
        Surface OpenRouter’s dynamic catalogue through the same interface as
        native providers so tools can reference OpenRouter models and aliases
        without special cases.

    Characteristics
        * Pulls live model definitions from :class:`OpenRouterModelRegistry`
          (aliases, provider-specific metadata, capability hints)
        * Applies alias-aware restriction checks before exposing models to the
          registry or tooling
        * Reuses :class:`OpenAICompatibleProvider` infrastructure for request
          execution so OpenRouter endpoints behave like standard OpenAI-style
          APIs.

    Batch execution
        ``submit_batch`` / ``poll_batch`` / ``fetch_batch`` / ``cancel_batch`` talk to
        OpenRouter's asynchronous Batch API (``{base_url}/batches``, i.e.
        ``/api/v1/batches``) directly over httpx. That is a separate resource from the
        chat-completions path the synchronous methods use, and nothing here touches the
        synchronous base URL or client.

        Model slug rule (one deterministic mapping, both spellings converge):

        1. A trailing ``:batch`` on the supplied identifier is stripped, so
           ``anthropic/claude-opus-4.5`` and ``anthropic/claude-opus-4.5:batch`` both name
           the base model. OpenRouter's ``:batch`` catalog entries are pricing listings; the
           batch envelope takes the model's *normal* slug.
        2. The remainder goes through the same alias resolution, capability lookup and
           restriction checks as a synchronous call (``get_capabilities`` /
           ``_resolve_model_name`` / ``_ensure_model_allowed``).
        3. A short *family alias* (``opus``, ``sonnet``, ``gemini``, ``flash``, ``gpt5`` ...: the
           aliases a curated entry's ``batch_family`` names) resolves DYNAMICALLY to the latest member
           of its family that OpenRouter lists as ``:batch`` (see ``providers/shared/batch_catalog.py``),
           after the restriction policy is applied to the candidates. Version-pinned aliases and full
           slugs never float.
        4. Any other resolved slug must be batch-eligible: OpenRouter's batch catalog lists it, or its
           curated entry has ``supports_batch`` (the offline fallback). A model that is neither, or that
           carries a variant (``:free``, ``:online``), is rejected locally, never remapped to another model.
        5. The resolved base slug is what is submitted; result metadata reports it as
           ``submitted_model`` and, when the caller's spelling differed, the caller's
           spelling as ``requested_model``.
        6. The limits applied to the call are those of the resolved slug itself: derived, per field, from its
           ``:batch`` listing (live catalog, then the committed snapshot), and only for a field neither has, from
           the curated family entry (one warning). A slug the curated catalog lists keeps its curated LIMITS, but
           the request PARAMETERS (``temperature``, ``top_p``, ``seed``, the token-cap spelling ...) are only those
           its ``:batch`` listing names, live then snapshot; with no listing at all the curated entry decides, with
           one warning. The whole result is a :class:`BatchModelResolution`, built once at admission and carried to
           submission.

        OpenRouter has no cancel endpoint, so ``cancel_batch`` reports ``unsupported`` for a
        running batch (it continues and is still billed) and never issues a DELETE.
    """

    FRIENDLY_NAME = "OpenRouter"

    # Custom headers required by OpenRouter
    DEFAULT_HEADERS = {
        "HTTP-Referer": get_env("OPENROUTER_REFERER", "https://github.com/izzoa/unison-mcp-server")
        or "https://github.com/izzoa/unison-mcp-server",
        "X-Title": get_env("OPENROUTER_TITLE", "Unison MCP Server") or "Unison MCP Server",
    }

    # Model registry for managing configurations and aliases
    _registry: OpenRouterModelRegistry | None = None

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            **kwargs: Additional configuration
        """
        base_url = "https://openrouter.ai/api/v1"
        self._alias_cache: dict[str, str] = {}
        # batch id -> identifier the caller supplied, so fetched results can report it.
        self._batch_requested_models: dict[str, str] = {}
        # batch id -> (slug actually submitted, submitted custom_ids). In-process only: a caller
        # that outlives the process persists the ids and passes them to fetch_batch.
        self._batch_submissions: dict[str, tuple[str, tuple[str, ...]]] = {}
        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if OpenRouterProvider._registry is None:
            OpenRouterProvider._registry = OpenRouterModelRegistry()
            # Log loaded models and aliases only on first load
            models = self._registry.list_models()
            aliases = self._registry.list_aliases()
            logging.info(f"OpenRouter loaded {len(models)} models with {len(aliases)} aliases")

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        """Fetch OpenRouter capabilities from the registry or build a generic fallback."""

        capabilities = self._registry.get_capabilities(canonical_name)
        if capabilities:
            return capabilities

        base_identifier, separator, _variant = canonical_name.partition(":")

        # The catalog lists a handful of models and none of their provider-published variants
        # (``:batch``, ``:free``, ``:online`` ...). A variant the catalog does not list inherits
        # its base model's capabilities rather than the generic defaults, which would understate
        # the context window and disable features the model supports. A catalogued variant entry
        # was already returned above and so takes precedence.
        if separator:
            inherited = self._inherit_variant_capabilities(base_identifier, canonical_name)
            if inherited is not None:
                return inherited

        if "/" in base_identifier:
            logging.debug(
                "Using generic OpenRouter capabilities for %s (provider/model format detected)", canonical_name
            )
            generic = ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name=canonical_name,
                friendly_name=self.FRIENDLY_NAME,
                intelligence_score=9,
                context_window=32_768,
                max_output_tokens=32_768,
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=False,
                temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            )
            generic._is_generic = True
            return generic

        logging.debug(
            "Rejecting unknown OpenRouter model '%s' (no provider prefix); requires explicit configuration",
            canonical_name,
        )
        return None

    def _inherit_variant_capabilities(self, base_identifier: str, variant_identifier: str) -> ModelCapabilities | None:
        """Return the base model's capabilities re-keyed to a variant identifier.

        Only a base that is itself a catalogued model name qualifies. An alias is not
        accepted: ``opus:batch`` is not an identifier OpenRouter serves, so it must not be
        given capabilities and then sent to the provider as if it were.
        """

        base = self._registry.get_capabilities(base_identifier)
        if base is None or base.model_name.lower() != base_identifier.lower():
            return None

        logging.debug(
            "Variant %s inherits capabilities from catalogued base model %s", variant_identifier, base.model_name
        )
        # The variant inherits what is published about its base (a colon identifier matches no LiteLLM key itself).
        base = apply_published_limits(ProviderType.OPENROUTER, base.model_name, base)
        return dataclasses.replace(base, model_name=variant_identifier, aliases=[])

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    def get_provider_type(self) -> ProviderType:
        """Identify this provider for restrictions and logging."""
        return ProviderType.OPENROUTER

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def list_models(
        self,
        *,
        respect_restrictions: bool = True,
        include_aliases: bool = True,
        lowercase: bool = False,
        unique: bool = False,
    ) -> list[str]:
        """Return formatted OpenRouter model names, respecting alias-aware restrictions."""

        if not self._registry:
            return []

        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        allowed_configs: dict[str, ModelCapabilities] = {}

        for model_name in self._registry.list_models():
            config = self._registry.resolve(model_name)
            if not config:
                continue

            # Custom models belong to CustomProvider; skip them here so the two
            # providers don't race over the same registrations (important for tests
            # that stub the registry with minimal objects lacking attrs).
            if config.provider == ProviderType.CUSTOM:
                continue

            if restriction_service:
                allowed = restriction_service.is_allowed(self.get_provider_type(), model_name)

                if not allowed and config.aliases:
                    for alias in config.aliases:
                        if restriction_service.is_allowed(self.get_provider_type(), alias):
                            allowed = True
                            break

                if not allowed:
                    continue

            allowed_configs[model_name] = config

        if not allowed_configs:
            return []

        # When restrictions are in place, don't include aliases to avoid confusion
        # Only return the canonical model names that are actually allowed
        actual_include_aliases = include_aliases and not respect_restrictions

        return ModelCapabilities.collect_model_names(
            allowed_configs,
            include_aliases=actual_include_aliases,
            lowercase=lowercase,
            unique=unique,
        )

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve aliases defined in the OpenRouter registry."""

        cache_key = model_name.lower()
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        config = self._registry.resolve(model_name)
        if config:
            if config.model_name != model_name:
                logging.debug("Resolved model alias '%s' to '%s'", model_name, config.model_name)
            resolved = config.model_name
            self._alias_cache[cache_key] = resolved
            self._alias_cache.setdefault(resolved.lower(), resolved)
            return resolved

        logging.debug(f"Model '{model_name}' not found in registry, using as-is")
        self._alias_cache[cache_key] = model_name
        return model_name

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        """Expose registry-backed OpenRouter capabilities."""

        if not self._registry:
            return {}

        capabilities: dict[str, ModelCapabilities] = {}
        for model_name in self._registry.list_models():
            config = self._registry.resolve(model_name)
            if not config:
                continue

            # See note in list_models: respect the CustomProvider boundary.
            if config.provider == ProviderType.CUSTOM:
                continue

            capabilities[model_name] = config
        return capabilities

    # ------------------------------------------------------------------
    # Batch execution (OpenRouter Batch API)
    # ------------------------------------------------------------------

    _BATCH_DEFAULT_TEMPERATURE = 0.3  # matches generate_content's default

    def _batch_catalog(self) -> BatchCatalog | None:
        """The batch catalog to resolve against now: live, else the committed snapshot, else ``None``.

        May fetch (bounded, cached, never raising); only batch admission and resolution call this, never a
        capability query.
        """

        return get_batch_catalog_source().current()

    def batch_resolution_scope(self) -> AbstractContextManager[None]:
        """Hold one catalog for a whole submission so every model in it resolves against the same data."""

        return get_batch_catalog_source().pin()

    def _floating_family(self, name: str) -> tuple[ModelCapabilities, BatchFamilyRule] | None:
        """The curated entry and rule when ``name`` is an alias that resolves dynamically under batch.

        A full slug never floats (even one that equals a curated ``model_name``), and neither does an
        alias the rule does not name (``opus4.5``, ``o3``, ``gpt5.2``).
        """

        config = self._registry.resolve(name)
        rule = config.batch_family if config is not None else None
        if config is None or rule is None:
            return None
        if config.model_name.lower() == name.strip().lower() or not rule.floats(name):
            return None
        return config, rule

    def _slug_allowed(self, slug: str, typed: str) -> bool:
        """Whether the restriction policy admits ``slug`` for a caller who typed ``typed``."""

        capabilities = self._lookup_capabilities(slug, typed)
        if capabilities is None:
            return False
        try:
            self._ensure_model_allowed(capabilities, slug, typed)
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_reviewed_entry(capabilities: ModelCapabilities | None, slug: str) -> bool:
        """True when ``capabilities`` is the reviewed ``conf/openrouter_models.json`` entry for exactly ``slug``.

        An entry the LiteLLM catalogue auto-discovered (``auto_discovered``) is not reviewed: it carries flags
        inferred for the synchronous path, and which slugs it covers depends on the installed LiteLLM version. It
        must not shadow the slug's own batch listing, so it is treated like no entry at all.
        """

        return (
            capabilities is not None
            and not capabilities.auto_discovered
            and capabilities.model_name.lower() == slug.strip().lower()
        )

    def _family_pick(self, typed: str, rule: BatchFamilyRule, catalog: BatchCatalog) -> str | None:
        """The latest batch-listed family member the restriction policy admits (restrictions first)."""

        members = [slug for slug in catalog.slugs if rule.is_member(slug) and self._slug_allowed(slug, typed)]
        return rule.select(members)

    def _family_rules(self) -> list[BatchFamilyRule]:
        """Every ``batch_family`` rule of every curated entry (what "a family rule classifies it" means)."""

        rules: list[BatchFamilyRule] = []
        for name in self._registry.list_models():
            config = self._registry.resolve(name)
            if config is not None and config.batch_family is not None:
                rules.append(config.batch_family)
        return rules

    @staticmethod
    def _listing_sources(slug: str, catalog: BatchCatalog | None) -> list[tuple[str, Any]]:
        """The listing views that speak for ``slug``, strongest first: the pinned catalog's ``:batch`` listing, the
        committed snapshot, then the plain listing of the live body.

        The catalog is the one the call pinned, so the limits and parameters come from the same data that chose
        the slug. A call whose catalog *is* the snapshot starts at the snapshot. ``catalog=None`` (not even the
        snapshot is readable) yields no source at all.
        """

        if catalog is None:
            return []
        exact, plain = catalog.view_layers(slug)
        sources: list[tuple[str, Any]] = [(catalog.origin, exact)]
        if catalog.origin != "snapshot":
            snapshot = get_batch_catalog_source().snapshot_catalog()
            sources.append(("snapshot", snapshot.view_for(slug) if snapshot is not None else None))
        if plain is not None:
            # Last: the plain listing of the live body is the weakest evidence for a batch limit (its numbers can
            # differ from the ``:batch`` listing's), so the snapshot's batch-derived value outranks it per field.
            sources.append((catalog.origin, plain))
        return sources

    def _derived_resolution(
        self,
        requested: str,
        slug: str,
        fallback: ModelCapabilities,
        *,
        fallback_label: str,
        catalog: BatchCatalog,
        floated: bool,
    ) -> BatchModelResolution:
        """Capabilities of ``slug`` from its own listing in ``catalog`` (live, then the snapshot), then ``fallback``.

        Per field; see :func:`resolve_capabilities`.
        """

        capabilities, where = resolve_capabilities(
            slug,
            sources=self._listing_sources(slug, catalog),
            fallback=fallback,
            fallback_label=fallback_label,
        )
        return BatchModelResolution(
            requested=requested,
            slug=slug,
            capabilities=capabilities,
            floated=floated,
            derived=True,
            catalog_origin=catalog.origin,
            sources=where,
        )

    def _curated_resolution(
        self,
        requested: str,
        slug: str,
        curated: ModelCapabilities,
        *,
        catalog: BatchCatalog | None,
        floated: bool,
    ) -> BatchModelResolution:
        """A reviewed curated entry: its LIMITS are kept, its request PARAMETERS come from the listing (D9).

        ``derived`` stays false, so the reviewed context window and output cap stay enforced. Which optional
        parameters a body carries is decided like a derived entry's (live ``:batch`` listing, then the snapshot,
        see :func:`apply_listed_parameters`); with no listing at all the curated entry decides, with one WARNING.
        """

        listings = self._listing_sources(slug, catalog)
        capabilities, source = apply_listed_parameters(curated, slug=slug, sources=listings)
        # D12/D16: the reviewed output cap and the temperature rules are bounded by what OpenRouter's listings and
        # LiteLLM publish for the slug (smallest valid figure; never raised above the reviewed one).
        capabilities, published = apply_published_bounds(capabilities, slug=slug, sources=listings, stand_in_cap=False)
        sources = {PARAMETERS_FIELD: source}
        if published is not None:
            sources[PUBLISHED_CAP_FIELD] = published
        return BatchModelResolution(
            requested=requested,
            slug=slug,
            capabilities=capabilities,
            floated=floated,
            catalog_origin=catalog.origin if catalog is not None else "",
            sources=sources,
        )

    def _unclassified_note(
        self, typed: str, picked: str, rule: BatchFamilyRule, catalog: BatchCatalog
    ) -> tuple[tuple[str, ...], int]:
        """Newer batch listings no family rule classifies, for the alias that picked ``picked`` (never raises).

        Computed after the pick, from the same catalog; it cannot change the pick. A failure yields no note.
        """

        try:
            names, omitted = unclassified_newer_listings(
                pick=picked,
                rule=rule,
                catalog=catalog,
                rules=self._family_rules(),
                # Each candidate is tested as the FULL slug the caller would have to type to use it: the note
                # advises exactly that, so a slug the policy refuses by its full name must not be advertised just
                # because the policy names the alias.
                allowed=lambda candidate: self._slug_allowed(candidate, candidate),
            )
        except Exception:
            logging.debug("Could not compute the unclassified-newer note for %s -> %s", typed, picked, exc_info=True)
            return (), 0
        if names:
            warn_once(
                ("unclassified", typed.lower(), picked, names, omitted),
                "Under batch, '%s' resolved to %s by version number, but OpenRouter also lists newer batch models "
                "that no family rule classifies, so the alias did not pick them: %s%s. Name a full slug to use one.",
                typed,
                picked,
                ", ".join(names),
                f" (and {omitted} more)" if omitted else "",
            )
        return names, omitted

    def _resolve_batch_model(
        self, model_name: str, catalog: BatchCatalog | None = _CATALOG_UNSET
    ) -> BatchModelResolution:
        """Apply the documented slug rule and return the :class:`BatchModelResolution`.

        ``catalog`` lets a caller that resolves several things in one operation hand in the one catalog
        it captured; left out, the source is asked once for this call.

        A family alias resolves to the latest batch-listed member of its family. The limits of the resolved slug
        are then, in this order and per field: its live ``:batch`` listing, the committed snapshot, the curated
        family entry (one warning). A full slug the curated catalog does not list but the catalog does gets the
        same chain with the generic defaults as its last resort. A slug the curated catalog lists keeps its curated
        limits, but the request parameters it sends are those its ``:batch`` listing names (live, then snapshot;
        the curated entry only when no listing exists). ``get_capabilities`` and the synchronous path never come
        through here.

        Raises:
            BatchModelRefused: no batch listing, a variant identifier, or a family with no batch member.
            ValueError: unknown or restricted model.
        """

        base = strip_batch_suffix(model_name)
        if not base:
            raise ValueError("A batch requires a model name")

        if catalog is _CATALOG_UNSET:
            catalog = self._batch_catalog()

        # A variant (``:free``, ``:online``, a doubled ``:batch`` ...) of a model OpenRouter serves is never
        # batch-listed, and stripping it would run a different model than the caller named, so it is refused
        # rather than normalised. A colon name OpenRouter does not serve at all (an Ollama or custom tag such
        # as ``llama3:8b``) is not this adapter's to explain: advice to "name the base model" would point at a
        # different model, so it is simply not one of ours.
        if ":" in base:
            if not self._serves_variant_base(base, catalog):
                raise ValueError(f"Model '{model_name}' is not served by OpenRouter")
            raise self._variant_refusal(model_name, base)

        floating = self._floating_family(base) if catalog is not None else None
        if floating is not None and catalog is not None:
            config, rule = floating
            picked = self._family_pick(base, rule, catalog)
            if picked is None:
                raise self._family_refusal(base, config, rule, catalog)
            curated = self._registry.get_capabilities(picked)
            if self._is_reviewed_entry(curated, picked):
                assert curated is not None
                resolution = self._curated_resolution(base, picked, curated, catalog=catalog, floated=True)
            else:
                resolution = self._derived_resolution(
                    base, picked, config, fallback_label="curated", catalog=catalog, floated=True
                )
            names, omitted = self._unclassified_note(base, picked, rule, catalog)
            return dataclasses.replace(resolution, unclassified=names, unclassified_omitted=omitted)

        # Alias resolution + restriction enforcement exactly as a synchronous call.
        capabilities = self.get_capabilities(base)
        submitted = self._resolve_model_name(base)
        if ":" in submitted:
            raise self._variant_refusal(model_name, submitted)

        # Eligible when OpenRouter's batch catalog lists the slug, or (the offline fallback) its curated
        # entry says so. The curated catalog is therefore not the ceiling: an uncurated slug the live list
        # carries as ``<slug>:batch`` batches too.
        catalogued = self._registry.get_capabilities(submitted)
        curated_exact = self._is_reviewed_entry(catalogued, submitted)
        committed = curated_exact and catalogued is not None and catalogued.supports_batch
        if not (committed or (catalog is not None and catalog.lists(submitted))):
            raise BatchModelRefused(
                f"Batch execution is unavailable for model '{model_name}' (resolved to '{submitted}'): "
                "OpenRouter has no batch listing for it. The same call without execution_mode runs it "
                "synchronously at standard pricing; or choose a batch-capable model."
            )
        if curated_exact or catalog is None:
            # A reviewed entry keeps its limits (a live body does not override them), but the parameters a request
            # carries are those its ``:batch`` listing names.
            return self._curated_resolution(base, submitted, capabilities, catalog=catalog, floated=False)
        return self._derived_resolution(
            base, submitted, capabilities, fallback_label="generic", catalog=catalog, floated=False
        )

    def _serves_variant_base(self, identifier: str, catalog: BatchCatalog | None) -> bool:
        """True when a colon identifier is one OpenRouter would own, so its variant is this adapter's to refuse.

        That is an identifier OpenRouter serves synchronously (``openai/gpt-5:free``), a variant of a slug the
        batch catalog lists, or a variant of a floating family alias (``opus:free``). A tag on a name OpenRouter
        does not serve (an Ollama ``llama3:8b``) is not one of ours, and advice to "name the base model" would
        point at a different model.
        """

        base_slug = identifier.partition(":")[0]
        if not base_slug:
            return False
        if self.validate_model_name(identifier):
            return True
        if catalog is not None and catalog.lists(base_slug):
            return True
        return self._floating_family(base_slug) is not None

    @staticmethod
    def _variant_refusal(model_name: str, identifier: str) -> BatchModelRefused:
        base_slug, _, variant = identifier.partition(":")
        return BatchModelRefused(
            f"Batch execution is unavailable for model '{model_name}': it carries the ':{variant}' variant, "
            "which has no batch listing, and OpenRouter's batch API takes the model's normal slug "
            f"('{base_slug}'). The variant is not stripped, because that would run a different model than the "
            "one named. The same call without execution_mode forwards the identifier unchanged; or name the base model."
        )

    @staticmethod
    def _family_refusal(
        alias: str, config: ModelCapabilities, rule: BatchFamilyRule, catalog: BatchCatalog
    ) -> BatchModelRefused:
        family = rule.label or config.model_name
        if any(rule.is_member(slug) for slug in catalog.slugs):
            why = (
                f"OpenRouter lists {family} batch models (per its {catalog.origin} catalog), but the restriction "
                "policy does not allow any of them: not allowed by restriction policy"
            )
        else:
            why = f"OpenRouter's {catalog.origin} catalog lists no batch model in the {family} family"
        return BatchModelRefused(
            f"Batch execution is unavailable for '{alias}': {why}, so nothing was chosen (a different family is "
            "never substituted). The same call without execution_mode runs it synchronously; or name a "
            "specific batch-capable model."
        )

    def resolve_batch_model(self, model_name: str) -> BatchModelResolution:
        """The resolution admission carries to planning and submission (see :meth:`_resolve_batch_model`).

        Call it inside :meth:`batch_resolution_scope` so the slug, its limits and the unclassified-newer note all
        come from the one pinned catalog. It may fetch that catalog (bounded, cached); a capability query never
        calls it.
        """

        return self._resolve_batch_model(model_name)

    def can_batch_model(self, model_name: str) -> bool:
        """True only when :meth:`submit_batch` would accept the identifier, by the same rule it applies.

        The capability flag alone is not enough: ``_lookup_capabilities`` lends a variant identifier
        (``:free``, ``:online``, ``:thinking``) its base model's flags, so ``supports_batch`` is ``True``
        for it while the adapter refuses it. Asking the adapter's own resolution keeps admission and
        submission in agreement.
        """

        try:
            self._resolve_batch_model(model_name)
        except Exception:
            return False
        return True

    def explain_batch_refusal(self, model_name: str) -> str | None:
        """Why this adapter deliberately refuses ``model_name`` under batch, or ``None``.

        ``None`` also for a model this provider does not serve at all (unknown or restricted): that is not
        a batch refusal, and a native provider may serve it.
        """

        try:
            self._resolve_batch_model(model_name)
        except BatchModelRefused as exc:
            return str(exc)
        except Exception:
            return None
        return None

    def _batch_slug(self, identifier: str, catalog: BatchCatalog | None = _CATALOG_UNSET) -> str:
        """Slug an identifier would be submitted as, without capability or eligibility checks.

        A family alias reports the member the catalog resolves it to (the same pick
        :meth:`_resolve_batch_model` makes, restrictions included), falling back to the curated pin when
        the catalog has no member to offer.
        """

        base = strip_batch_suffix(identifier)
        if catalog is _CATALOG_UNSET:
            catalog = self._batch_catalog()
        floating = self._floating_family(base) if catalog is not None else None
        if floating is not None and catalog is not None:
            picked = self._family_pick(base, floating[1], catalog)
            if picked is not None:
                return picked
        return self._resolve_model_name(base)

    def batch_family_preview(self, name: str) -> str | None:
        """What a family alias would resolve to under batch from data already loaded (never fetches).

        For ``listmodels``: ``None`` when ``name`` is not a floating alias or nothing is known to pick from.
        """

        catalog = get_batch_catalog_source().loaded()
        floating = self._floating_family(name) if catalog is not None else None
        if floating is None or catalog is None:
            return None
        return self._family_pick(name, floating[1], catalog)

    def _reject_mixed_models(
        self,
        requests: list[BatchRequest],
        resolution: BatchModelResolution,
        catalog: BatchCatalog | None = _CATALOG_UNSET,
    ) -> None:
        """A batch carries one model; reject any request that names a different one.

        A request that names what the resolution was made for (the alias typed, or the slug it resolved to) is the
        batch's model by construction; any other name is resolved the way a submission would resolve it, against
        ``catalog`` (asked for lazily, so a resolved batch whose requests all agree never touches the catalog).
        """

        known = {resolution.requested.lower(), resolution.slug.lower()}
        for request in requests:
            named = request.extra.get("model")
            if not named:
                continue
            if strip_batch_suffix(str(named)).lower() in known:
                continue
            if catalog is _CATALOG_UNSET:
                catalog = self._batch_catalog()
            if self._batch_slug(str(named), catalog).lower() != resolution.slug.lower():
                raise ValueError(
                    f"A batch carries one model: request '{request.custom_id}' names '{named}' but the "
                    f"batch model is '{resolution.slug}'. Submit one batch per model."
                )

    def _batch_send(self, method: str, path: str, content: str | None = None) -> httpx.Response:
        """Issue one batch-API HTTP call. The single seam tests stub to avoid the network.

        ``path`` is relative to ``base_url`` (for example ``/batches``). Timeouts and proxy
        handling match the synchronous client; ``_test_transport`` is honoured the same way.
        """

        url = f"{(self.base_url or '').rstrip('/')}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            **self.DEFAULT_HEADERS,
        }
        if content is not None:
            headers["Content-Type"] = "application/json"

        timeout = self.timeout_config if getattr(self, "timeout_config", None) else httpx.Timeout(30.0)
        proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
        client_kwargs: dict = {"timeout": timeout, "follow_redirects": True}
        if hasattr(self, "_test_transport"):
            client_kwargs["transport"] = self._test_transport

        with suppress_env_vars(*proxy_env_vars):
            with httpx.Client(**client_kwargs) as http:
                return http.request(method, url, headers=headers, content=content.encode("utf-8") if content else None)

    def _batch_get_json(self, batch_id: str, *, action: str) -> dict:
        """GET one batch object, mapping HTTP and transport failures to typed errors."""

        try:
            response = self._batch_send("GET", f"/batches/{quote(batch_id, safe='')}")
        except httpx.TransportError as exc:
            raise BatchProviderError(
                f"OpenRouter batch {action} could not reach the provider: {exc}",
                provider="openrouter",
                retryable=True,
            ) from exc

        raise_for_http_error(response.status_code, response.headers, response.text, action=action)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BatchProviderError(
                f"OpenRouter batch {action} returned a non-JSON body", provider="openrouter", retryable=True
            ) from exc
        if not isinstance(payload, dict):
            raise BatchProviderError(f"OpenRouter batch {action} returned an unexpected body", provider="openrouter")
        return payload

    @staticmethod
    def _bound_output_request(request: BatchRequest, resolution: BatchModelResolution) -> BatchRequest:
        """``request`` with its ``max_output_tokens`` reduced to the model's stated output cap (D12, D16).

        The cap is the smallest valid figure among OpenRouter's listings, LiteLLM and the reviewed entry (see
        :meth:`BatchModelResolution.output_cap_bound`), so a request is never asked to exceed any published limit.
        One WARNING per model and cap per process names the reduction. With nothing stated (a stand-in cap only) the
        caller's figure goes out as given, with one DEBUG line: no limit is invented.
        """

        requested = request.max_output_tokens
        if not requested:
            return request
        bound = resolution.output_cap_bound()
        if bound is None:
            warn_once(
                ("output-unbounded", resolution.slug),
                "No published output limit exists for batch model %s (no listing, no LiteLLM entry, no reviewed cap), "
                "so a requested max_output_tokens is sent as given",
                resolution.slug,
            )
            return request
        cap, source = bound
        if requested <= cap:
            return request
        warn_once(
            ("output-clamp", resolution.slug, cap),
            "Batch model %s: max_output_tokens %d is above the %s output limit (%d) and is reduced to it",
            resolution.slug,
            requested,
            source,
            cap,
        )
        return dataclasses.replace(request, max_output_tokens=cap)

    def submit_batch(
        self, model_name: str, requests: list[BatchRequest], resolution: BatchModelResolution | None = None
    ) -> str:
        """Submit ``requests`` as one inline batch and return OpenRouter's batch id.

        Every rejection (duplicate ids, non-text content, a model without a batch listing,
        a restricted model, a second model) happens before any network call. The submission
        is never retried here: a failed POST may or may not have created a batch, which is
        for the caller's saga to reconcile. Acceptance (``202``, ``validating``) means queued,
        not complete.

        ``resolution`` is what admission resolved ``model_name`` to; the batch is submitted under exactly that slug
        and those limits (no second resolution, so a catalog that moved since admission changes neither). Without
        one (a direct call) the name is resolved here, once, against one catalog.
        """

        requests = list(requests)
        validate_batch_requests(requests)
        reject_non_text(requests)
        reject_invalid_temperature(requests)
        catalog: Any = _CATALOG_UNSET
        if resolution is None:
            # One catalog for everything this call resolves, so the slug checked, compared and sent cannot differ.
            catalog = self._batch_catalog()
            resolution = self._resolve_batch_model(model_name, catalog)
        elif resolution.requested.lower() != strip_batch_suffix(model_name).lower():
            raise ValueError(
                f"The batch resolution is for '{resolution.requested}', not '{model_name}'; resolve the model "
                "this batch is for."
            )
        submitted, capabilities = resolution.slug, resolution.capabilities
        self._reject_mixed_models(requests, resolution, catalog)

        entries = []
        for request in requests:
            temperature = request.temperature if request.temperature is not None else self._BATCH_DEFAULT_TEMPERATURE
            effective = capabilities.get_effective_temperature(temperature)
            if effective is not None and not capabilities.temperature_constraint.validate(effective):
                raise ValueError(
                    f"Temperature {effective} is invalid for model {submitted}. "
                    f"{capabilities.temperature_constraint.get_description()}"
                )
            listed = capabilities.supported_parameters
            request = self._bound_output_request(request, resolution)
            entries.append(
                (
                    request.custom_id,
                    build_chat_body(
                        request,
                        temperature=effective,
                        include_max_tokens=effective is not None,
                        listed_parameters=listed,
                    ),
                )
            )

        payload = serialize_envelope(submitted, entries)
        try:
            response = self._batch_send("POST", "/batches", content=payload)
        except httpx.TransportError as exc:
            # The request may have reached OpenRouter before the connection failed, so a batch
            # may exist. Typed, and not retryable here: the caller must reconcile, not resend.
            raise BatchProviderError(
                f"OpenRouter batch submit could not confirm the outcome ({type(exc).__name__}: {exc}); "
                "a batch may or may not have been created. Reconcile before resubmitting.",
                provider="openrouter",
                code=SUBMISSION_OUTCOME_UNKNOWN,
            ) from exc
        raise_for_http_error(response.status_code, response.headers, response.text, action="submit")

        try:
            body = response.json()
        except ValueError:
            body = None
        batch_id = body.get("id") if isinstance(body, dict) else None
        if not isinstance(batch_id, str) or not batch_id:
            # Accepted, so a billable batch exists that nobody can name: ambiguous, not a refusal.
            raise BatchProviderError(
                f"OpenRouter accepted the batch submission (HTTP {response.status_code}) but returned no batch id; "
                "a batch exists that cannot be named. Reconcile before resubmitting.",
                provider="openrouter",
                status_code=response.status_code,
                code=SUBMISSION_OUTCOME_UNKNOWN,
            )

        self._batch_requested_models[batch_id] = model_name
        self._batch_submissions[batch_id] = (submitted, tuple(request.custom_id for request in requests))
        return batch_id

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Report a batch's state. ``validating`` and other in-flight states are non-terminal.

        Raises:
            BatchRateLimited / BatchProviderError(retryable=True): HTTP 429, 5xx or transport
                failure, which a wait loop may absorb and retry.
            BatchNotFound: unknown batch id (HTTP 404).
        """

        return parse_status(self._batch_get_json(batch_id, action="poll"))

    def fetch_batch(self, batch_id: str, expected_custom_ids: Sequence[str] | None = None) -> list[BatchOutcome]:
        """Return one outcome per request, correlated on ``custom_id`` (never on order).

        ``expected_custom_ids`` defaults to the ids this process submitted for ``batch_id``.
        With ids, every one gets exactly one outcome (a request OpenRouter returned nothing
        for is a failed ``missing_result`` outcome). With neither, a result set shorter than
        ``request_counts.total`` raises ``BatchIncompleteResults``.

        Raises:
            BatchNotReady: the batch is not terminal yet; poll first.
            BatchFailed: the batch ended without results.
            BatchResultsExpired: HTTP 410, results are past the 30-day retention.
            BatchIncompleteResults: fewer results than requests and no ids to attribute it to.
        """

        payload = self._batch_get_json(batch_id, action="fetch")
        submission = self._batch_submissions.get(batch_id)
        if expected_custom_ids is None and submission is not None:
            expected_custom_ids = submission[1]
        return parse_results(
            payload,
            batch_id=batch_id,
            requested_model=self._batch_requested_models.get(batch_id),
            friendly_name=self.FRIENDLY_NAME,
            submitted_model=submission[0] if submission else None,
            expected_custom_ids=expected_custom_ids,
        )

    def batch_submission_model(self, model_name: str) -> str:
        """The slug ``submit_batch`` would put in the envelope, without eligibility checks."""

        return self._batch_slug(model_name)

    def list_batches(self, limit: int = 100) -> BatchListing:
        """List the caller's recent batches: ``GET /batches?limit=...`` (metadata only).

        Used by reconciliation to correlate an unrecorded submission with a listed batch.
        ``results`` is always ``null`` in the listing and a running batch's ``custom_id``s
        are not visible, so correlation is approximate. A full page is flagged as possibly
        truncated. 429/5xx/transport failures raise retryable ``BatchProviderError``.

        OpenRouter accepts ``limit`` from 1 through 100 and answers an invalid query with a
        400, so the value is clamped to that range.
        """

        limit = max(1, min(int(limit), 100))
        try:
            response = self._batch_send("GET", f"/batches?limit={limit}")
        except httpx.TransportError as exc:
            raise BatchProviderError(
                f"OpenRouter batch list could not reach the provider: {exc}",
                provider="openrouter",
                retryable=True,
            ) from exc
        raise_for_http_error(response.status_code, response.headers, response.text, action="list")
        try:
            payload = response.json()
        except ValueError as exc:
            raise BatchProviderError(
                "OpenRouter batch list returned a non-JSON body", provider="openrouter", retryable=True
            ) from exc
        return parse_listing(payload, limit=limit)

    def cancel_batch(self, batch_id: str) -> CancelOutcome:
        """Report honestly: OpenRouter has no cancel endpoint, so a running batch continues.

        A terminal batch answers ``already_terminal``; a running one answers ``unsupported``.
        Neither raises and no DELETE is ever issued (OpenRouter refuses it for a running
        batch with 409, and deleting finished work is not a cancel).
        """

        status = self.poll_batch(batch_id)
        if status.is_terminal:
            return CancelOutcome(
                CancelState.ALREADY_TERMINAL,
                f"OpenRouter batch {batch_id} already finished as '{status.state.value}'; nothing to cancel.",
            )
        return CancelOutcome(
            CancelState.UNSUPPORTED,
            f"OpenRouter has no cancel endpoint. Batch {batch_id} is '{status.state.value}', will continue "
            "running, and is still billed. Polling can simply stop.",
        )
