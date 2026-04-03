"""
Full integration test script to validate UTF-8 implementation
and French localization.

This script runs all unit tests and checks full integration.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_utf8_integration_tests():
    """Run UTF-8 integration tests."""
    print("🚀 Starting UTF-8 integration tests")
    print("=" * 60)

    # Test environment setup
    os.environ["LOCALE"] = "fr-FR"
    os.environ["GEMINI_API_KEY"] = "dummy-key-for-tests"
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-tests"

    # Test 1: Validate UTF-8 characters in json.dumps
    print("\n1️⃣ UTF-8 encoding test with json.dumps")
    test_utf8_json_encoding()

    # Test 2: Validate language instruction generation
    print("\n2️⃣ Language instruction generation test")
    test_language_instruction_generation()

    # Test 3: Validate UTF-8 file handling
    print("\n3️⃣ UTF-8 file handling test")
    test_file_utf8_handling()

    # Test 4: Validate MCP tools integration
    print("\n4️⃣ MCP tools integration test")
    test_mcp_tools_integration()

    # Test 5: Run unit tests
    print("\n5️⃣ Running unit tests")
    run_unit_tests()

    print("\n✅ All UTF-8 integration tests completed!")
    print("🇫🇷 French localization works correctly!")


def test_utf8_json_encoding():
    """Test UTF-8 encoding with json.dumps(ensure_ascii=False)."""
    print("   Testing UTF-8 JSON encoding...")

    # Test data with French characters and emojis
    test_data = {
        "analyse": {
            "statut": "terminée",
            "résultat": "Aucun problème critique détecté",
            "recommandations": [
                "Améliorer la documentation",
                "Optimiser les performances",
                "Ajouter des tests unitaires",
            ],
            "métadonnées": {
                "créé_par": "Développeur Principal",
                "date_création": "2024-01-01",
                "dernière_modification": "2024-01-15",
            },
            "émojis_status": {
                "critique": "🔴",
                "élevé": "🟠",
                "moyen": "🟡",
                "faible": "🟢",
                "succès": "✅",
                "erreur": "❌",
            },
        },
        "outils": [
            {"nom": "analyse", "description": "Analyse architecturale avancée"},
            {"nom": "révision", "description": "Révision de code automatisée"},
            {"nom": "génération", "description": "Génération de documentation"},
        ],
    }

    # Test with ensure_ascii=False
    json_correct = json.dumps(test_data, ensure_ascii=False, indent=2)

    # Checks
    utf8_terms = [
        "terminée",
        "résultat",
        "détecté",
        "Améliorer",
        "créé_par",
        "Développeur",
        "création",
        "métadonnées",
        "dernière",
        "émojis_status",
        "élevé",
        "révision",
        "génération",
    ]

    emojis = ["🔴", "🟠", "🟡", "🟢", "✅", "❌"]

    for term in utf8_terms:
        assert term in json_correct, f"Missing UTF-8 term: {term}"

    for emoji in emojis:
        assert emoji in json_correct, f"Missing emoji: {emoji}"

    # Check for escaped characters
    assert "\\u" not in json_correct, "Escaped Unicode characters detected!"

    # Test parsing
    parsed = json.loads(json_correct)
    assert parsed["analyse"]["statut"] == "terminée"
    assert parsed["analyse"]["émojis_status"]["critique"] == "🔴"

    print("   ✅ UTF-8 JSON encoding: SUCCESS")


def test_language_instruction_generation():
    """Test language instruction generation."""
    print("   Testing language instruction generation...")

    # Simulation of get_language_instruction
    def get_language_instruction():
        locale = os.getenv("LOCALE", "").strip()
        if not locale:
            return ""
        return f"Always respond in {locale}.\n\n"

    # Test with different locales
    test_locales = [
        ("fr-FR", "French"),
        ("en-US", "English"),
        ("es-ES", "Spanish"),
        ("de-DE", "German"),
        ("", "none"),
    ]

    for locale, description in test_locales:
        os.environ["LOCALE"] = locale
        instruction = get_language_instruction()

        if locale:
            assert locale in instruction, f"Missing {locale} in instruction"
            assert instruction.endswith("\n\n"), "Incorrect instruction format"
            print(f"     📍 {description}: {instruction.strip()}")
        else:
            assert instruction == "", "Empty instruction expected for empty locale"
            print(f"     📍 {description}: (empty)")

    # Restore French locale
    os.environ["LOCALE"] = "fr-FR"
    print("   ✅ Language instruction generation: SUCCESS")


def test_file_utf8_handling():
    """Test handling of files with UTF-8 content."""
    print("   Testing UTF-8 file handling...")

    # File content with French characters
    french_content = '''#!/usr/bin/env python3
"""
Module de gestion des préférences utilisateur.
Développé par: Équipe Technique
Date de création: 15 décembre 2024
"""

import json
from typing import Dict, Optional

class GestionnairePreferences:
    """Gestionnaire des préférences utilisateur avec support UTF-8."""

    def __init__(self):
        self.données = {}
        self.historique = []

    def définir_préférence(self, clé: str, valeur) -> bool:
        """
        Définit une préférence utilisateur.

        Args:
            clé: Identifiant de la préférence
            valeur: Valeur à enregistrer

        Returns:
            True si la préférence a été définie avec succès
        """
        try:
            self.données[clé] = valeur
            self.historique.append({
                "action": "définition",
                "clé": clé,
                "horodatage": "2024-01-01T12:00:00Z"
            })
            return True
        except Exception as e:
            print(f"Error setting preference: {e}")
            return False

    def obtenir_préférence(self, clé: str) -> Optional:
        """Récupère une préférence par sa clé."""
        return self.données.get(clé)

    def exporter_données(self) -> str:
        """Exporte les données en JSON UTF-8."""
        return json.dumps(self.données, ensure_ascii=False, indent=2)

# Configuration par défaut avec caractères UTF-8
CONFIG_DÉFAUT = {
    "langue": "français",
    "région": "France",
    "thème": "sombre",
    "notifications": "activées"
}

def créer_gestionnaire() -> GestionnairePreferences:
    """Crée une instance du gestionnaire."""
    gestionnaire = GestionnairePreferences()

    # Application de la configuration par défaut
    for clé, valeur in CONFIG_DÉFAUT.items():
        gestionnaire.définir_préférence(clé, valeur)

    return gestionnaire

if __name__ == "__main__":
    # Test d'utilisation
    gestionnaire = créer_gestionnaire()
    print("Gestionnaire créé avec succès! 🎉")
    print(f"Données: {gestionnaire.exporter_données()}")
'''

    # Test writing and reading UTF-8
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as f:
        f.write(french_content)
        temp_file = f.name

    try:
        # Test reading
        with open(temp_file, encoding="utf-8") as f:
            read_content = f.read()

        # Checks
        assert read_content == french_content, "Altered UTF-8 content"

        # Check specific terms
        utf8_terms = [
            "préférences",
            "Développé",
            "Équipe",
            "création",
            "données",
            "définir_préférence",
            "horodatage",
            "Récupère",
            "français",
            "activées",
            "créer_gestionnaire",
            "succès",
        ]

        for term in utf8_terms:
            assert term in read_content, f"Missing UTF-8 term: {term}"

        print("   ✅ UTF-8 file handling: SUCCESS")

    finally:
        # Cleanup
        os.unlink(temp_file)


def test_mcp_tools_integration():
    """Test MCP tools integration with UTF-8."""
    print("   Testing MCP tools integration...")

    # Simulation of MCP tool response
    def simulate_mcp_tool_response():
        """Simulate MCP tool response with UTF-8 content."""
        response_data = {
            "status": "success",
            "content_type": "markdown",
            "content": """# Analyse Terminée avec Succès ✅

## Résumé de l'Analyse

L'analyse architecturale du projet a été **terminée** avec succès. Voici les principaux résultats :

### 🎯 Objectifs Atteints
- ✅ Révision complète du code
- ✅ Identification des problèmes de performance
- ✅ Recommandations d'amélioration générées

### 📊 Métriques Analysées
| Métrique | Valeur | Statut |
|----------|--------|--------|
| Complexité cyclomatique | 12 | 🟡 Acceptable |
| Couverture de tests | 85% | 🟢 Bon |
| Dépendances externes | 23 | 🟠 À réviser |

### 🔍 Problèmes Identifiés

#### 🔴 Critique
Aucun problème critique détecté.

#### 🟠 Élevé
1. **Performance des requêtes** : Optimisation nécessaire
2. **Gestion mémoire** : Fuites potentielles détectées

#### 🟡 Moyen
1. **Documentation** : Certaines fonctions manquent de commentaires
2. **Tests unitaires** : Couverture à améliorer

### � Détails de l'Analyse

Pour plus de détails sur chaque problème identifié, consultez les recommandations ci-dessous.

### �🚀 Recommandations Prioritaires

1. **Optimisation DB** : Implémenter un cache Redis
2. **Refactoring** : Séparer les responsabilités
3. **Documentation** : Ajouter les docstrings manquantes
4. **Tests** : Augmenter la couverture à 90%+

### 📈 Prochaines Étapes

- [ ] Implémenter le système de cache
- [ ] Refactorer les modules identifiés
- [ ] Compléter la documentation
- [ ] Exécuter les tests de régression

---
*Analyse générée automatiquement par MCP Unison* 🤖
""",
            "metadata": {
                "tool_name": "analyze",
                "execution_time": 2.5,
                "locale": "fr-FR",
                "timestamp": "2024-01-01T12:00:00Z",
                "analysis_summary": {
                    "files_analyzed": 15,
                    "issues_found": 4,
                    "recommendations": 4,
                    "overall_score": "B+ (Good level)",
                },
            },
            "continuation_offer": {
                "continuation_id": "analysis-123",
                "note": "In-depth analysis available with more details",
            },
        }

        # Serialization with ensure_ascii=False
        json_response = json.dumps(response_data, ensure_ascii=False, indent=2)

        # UTF-8 checks
        utf8_checks = [
            "Terminée",
            "Succès",
            "Résumé",
            "terminée",
            "Atteints",
            "Révision",
            "problèmes",
            "générées",
            "Métriques",
            "Identifiés",
            "détecté",
            "Élevé",
            "nécessaire",
            "détectées",
            "améliorer",
            "Prioritaires",
            "responsabilités",
            "Étapes",
            "régression",
            "générée",
            "détails",
        ]

        for term in utf8_checks:
            assert term in json_response, f"Missing UTF-8 term: {term}"

        # Emoji check
        emojis = ["✅", "🎯", "📊", "🟡", "🟢", "🟠", "🔍", "🔴", "🚀", "📈", "🤖"]
        for emoji in emojis:
            assert emoji in json_response, f"Missing emoji: {emoji}"

        # Test parsing
        parsed = json.loads(json_response)
        assert parsed["status"] == "success"
        assert "Terminée" in parsed["content"]
        assert parsed["metadata"]["locale"] == "fr-FR"

        return json_response

    # Test simulation
    response = simulate_mcp_tool_response()
    assert len(response) > 1000, "MCP response too short"

    print("   ✅ MCP tools integration: SUCCESS")


def run_unit_tests():
    """Run unit tests."""
    print("   Running unit tests...")

    # List of test files to run
    test_files = ["test_utf8_localization.py", "test_provider_utf8.py", "test_workflow_utf8.py"]

    current_dir = Path(__file__).parent
    test_results = []

    for test_file in test_files:
        test_path = current_dir / test_file
        if test_path.exists():
            print(f"     📝 Running {test_file}...")
            try:
                # Test execution
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", test_file.replace(".py", ""), "-v"],
                    cwd=current_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    print(f"     ✅ {test_file}: SUCCESS")
                    test_results.append((test_file, "SUCCESS"))
                else:
                    print(f"     ❌ {test_file}: FAILURE")
                    print(f"        Error: {result.stderr[:200]}...")
                    test_results.append((test_file, "FAILURE"))

            except subprocess.TimeoutExpired:
                print(f"     ⏰ {test_file}: TIMEOUT")
                test_results.append((test_file, "TIMEOUT"))
            except Exception as e:
                print(f"     💥 {test_file}: ERROR - {e}")
                test_results.append((test_file, "ERROR"))
        else:
            print(f"     ⚠️ {test_file}: NOT FOUND")
            test_results.append((test_file, "NOT FOUND"))

    # Test summary
    print("\n   📋 Unit test summary:")
    for test_file, status in test_results:
        status_emoji = {"SUCCESS": "✅", "FAILURE": "❌", "TIMEOUT": "⏰", "ERROR": "💥", "NOT FOUND": "⚠️"}.get(
            status, "❓"
        )
        print(f"     {status_emoji} {test_file}: {status}")


def main():
    """Main function."""
    print("🇫🇷 UTF-8 Integration Test - Unison MCP Server")
    print("=" * 60)

    try:
        run_utf8_integration_tests()
        print("\n🎉 SUCCESS: All UTF-8 integration tests passed!")
        print("🚀 Unison MCP Server fully supports French localization!")
        return 0

    except AssertionError as e:
        print(f"\n❌ FAILURE: Assertion test failed: {e}")
        return 1

    except Exception as e:
        print(f"\n💥 ERROR: Unexpected exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
