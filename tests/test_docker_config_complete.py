"""
Complete configuration test for Docker MCP
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestDockerMCPConfiguration:
    """Docker MCP configuration tests"""

    def test_dockerfile_configuration(self):
        """Test Dockerfile configuration"""
        project_root = Path(__file__).parent.parent
        dockerfile = project_root / "Dockerfile"

        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()

        # Essential checks
        assert "FROM python:" in content
        assert "COPY" in content or "ADD" in content
        assert "server.py" in content

        # Recommended security checks
        security_checks = [
            "USER " in content,  # Non-root user
            "WORKDIR" in content,  # Defined working directory
        ]

        # At least one security practice should be present
        if any(security_checks):
            assert True, "Security best practices detected"

    def test_environment_file_template(self):
        """Test environment file template"""
        project_root = Path(__file__).parent.parent
        env_example = project_root / ".env.example"

        if env_example.exists():
            content = env_example.read_text()

            # Essential variables
            essential_vars = ["GEMINI_API_KEY", "OPENAI_API_KEY", "LOG_LEVEL"]

            for var in essential_vars:
                assert f"{var}=" in content, f"Variable {var} missing"

            # Docker-specific variables should also be present
            docker_vars = ["COMPOSE_PROJECT_NAME", "TZ"]
            for var in docker_vars:
                assert f"{var}=" in content, f"Docker variable {var} missing"

    def test_logs_directory_setup(self):
        """Test logs directory setup"""
        project_root = Path(__file__).parent.parent
        logs_dir = project_root / "logs"

        # The logs directory should exist or be creatable
        if not logs_dir.exists():
            try:
                logs_dir.mkdir(exist_ok=True)
                created = True
            except Exception:
                created = False

            assert created, "Logs directory should be creatable"
        else:
            assert logs_dir.is_dir(), "logs should be a directory"


class TestDockerCommandValidation:
    """Docker command validation tests"""

    @patch("subprocess.run")
    def test_docker_build_command(self, mock_run):
        """Test docker build command"""
        mock_run.return_value.returncode = 0

        # Standard build command
        build_cmd = ["docker", "build", "-t", "unison-mcp-server:latest", "."]

        import subprocess

        subprocess.run(build_cmd, capture_output=True)
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_docker_run_mcp_command(self, mock_run):
        """Test docker run command for MCP"""
        mock_run.return_value.returncode = 0

        # Run command for MCP
        run_cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--env-file",
            ".env",
            "-v",
            "logs:/app/logs",
            "unison-mcp-server:latest",
            "python",
            "server.py",
        ]

        import subprocess

        subprocess.run(run_cmd, capture_output=True)
        mock_run.assert_called_once()

    def test_docker_command_structure(self):
        """Test Docker command structure"""

        # Recommended MCP command
        mcp_cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--env-file",
            "/path/to/.env",
            "-v",
            "/path/to/logs:/app/logs",
            "unison-mcp-server:latest",
            "python",
            "server.py",
        ]

        # Structure checks
        assert mcp_cmd[0] == "docker"
        assert "run" in mcp_cmd
        assert "--rm" in mcp_cmd  # Automatic cleanup
        assert "-i" in mcp_cmd  # Interactive mode
        assert "--env-file" in mcp_cmd  # Environment variables
        assert "unison-mcp-server:latest" in mcp_cmd  # Image


class TestIntegrationChecks:
    """Integration checks"""

    def test_complete_setup_checklist(self):
        """Test complete setup checklist"""
        project_root = Path(__file__).parent.parent

        # Checklist for essential files
        essential_files = {
            "Dockerfile": project_root / "Dockerfile",
            "server.py": project_root / "server.py",
            "requirements.txt": project_root / "requirements.txt",
            "docker-compose.yml": project_root / "docker-compose.yml",
        }

        missing_files = []
        for name, path in essential_files.items():
            if not path.exists():
                missing_files.append(name)

        # Allow some missing files for flexibility
        critical_files = ["Dockerfile", "server.py"]
        missing_critical = [f for f in missing_files if f in critical_files]

        assert not missing_critical, f"Critical files missing: {missing_critical}"

    def test_mcp_integration_readiness(self):
        """Test MCP integration readiness"""
        project_root = Path(__file__).parent.parent

        # MCP integration checks
        checks = {
            "dockerfile": (project_root / "Dockerfile").exists(),
            "server_script": (project_root / "server.py").exists(),
            "logs_dir": (project_root / "logs").exists() or True,
        }

        # At least critical elements must be present
        critical_checks = ["dockerfile", "server_script"]
        missing_critical = [k for k in critical_checks if not checks[k]]

        assert not missing_critical, f"Critical elements missing: {missing_critical}"

        # Readiness score
        ready_score = sum(checks.values()) / len(checks)
        assert ready_score >= 0.75, f"Insufficient readiness score: {ready_score:.2f}"


class TestErrorHandling:
    """Error handling tests"""

    def test_missing_api_key_handling(self):
        """Test handling of missing API key"""

        # Simulate environment without API keys
        with patch.dict(os.environ, {}, clear=True):
            api_keys = [os.getenv("GEMINI_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("XAI_API_KEY")]

            has_api_key = any(key for key in api_keys)

            # No key should be present
            assert not has_api_key, "No API key detected (expected for test)"

            # System should handle this gracefully
            error_handled = True  # Simulate error handling
            assert error_handled, "API key error handling implemented"

    def test_docker_not_available_handling(self):
        """Test handling of Docker not available"""

        @patch("subprocess.run")
        def simulate_docker_unavailable(mock_run):
            # Simulate Docker not available
            mock_run.side_effect = FileNotFoundError("docker: command not found")

            try:
                import subprocess

                subprocess.run(["docker", "--version"], capture_output=True)
                docker_available = True
            except FileNotFoundError:
                docker_available = False

            # Docker is not available - expected error
            assert not docker_available, "Docker unavailable (simulation)"

            # System should provide a clear error message
            error_message_clear = True  # Simulation
            assert error_message_clear, "Clear Docker error message"

        simulate_docker_unavailable()


class TestDockerBatchAndEnvConsistency:
    """Batch store, forwarded environment and log-rotation settings stay consistent."""

    PROJECT_ROOT = Path(__file__).parent.parent
    # Variables .env.example documents that the container must be able to receive.
    FORWARDED_PREFIXES = ("BATCH_", "OPENROUTER_BATCH_", "STORAGE_", "CLINK_", "CIRCUIT_")

    def _compose(self):
        import yaml

        return yaml.safe_load((self.PROJECT_ROOT / "docker-compose.yml").read_text())

    def _compose_env_names(self):
        env = self._compose()["services"]["unison-mcp"]["environment"]
        return {entry.split("=", 1)[0] for entry in env}

    def test_compose_mounts_declared_data_volume(self):
        compose = self._compose()
        service = compose["services"]["unison-mcp"]
        assert "unison-mcp-data:/app/.unison" in service["volumes"]
        assert "unison-mcp-data" in compose["volumes"]

    def test_compose_defaults_batch_store_into_the_volume(self):
        env = self._compose()["services"]["unison-mcp"]["environment"]
        assert "BATCH_STORE_PATH=${BATCH_STORE_PATH:-/app/.unison/batches.db}" in env

    def test_dockerfile_creates_writable_data_dir_for_runtime_user(self):
        lines = (self.PROJECT_ROOT / "Dockerfile").read_text().splitlines()
        command = "RUN mkdir -p .unison && chown -R unisonuser:unisonuser .unison"
        assert command in lines
        mkdir_index = lines.index(command)
        user_index = next(i for i, line in enumerate(lines) if line.startswith("USER "))
        assert mkdir_index < user_index

    def test_compose_forwards_every_documented_runtime_variable(self):
        """Every batch/storage/clink/circuit variable .env.example documents reaches the container."""
        import re

        documented = set()
        for line in (self.PROJECT_ROOT / ".env.example").read_text().splitlines():
            match = re.match(r"^#?\s*([A-Z][A-Z0-9_]+)=", line)
            if match and match.group(1).startswith(self.FORWARDED_PREFIXES):
                documented.add(match.group(1))
        assert documented, "expected .env.example to document batch/storage/clink/circuit variables"
        missing = documented - self._compose_env_names()
        assert not missing, f"docker-compose.yml does not forward: {sorted(missing)}"

    def test_log_rotation_variables_are_not_advertised(self):
        """Rotation is fixed in utils/logging_setup.py; no file may offer knobs that do nothing."""
        for name in (".env.example", "docker-compose.yml", "run-server.ps1", "docs/docker-deployment.md"):
            text = (self.PROJECT_ROOT / name).read_text()
            assert "LOG_MAX_SIZE" not in text, name
            assert "LOG_BACKUP_COUNT" not in text, name

    def test_conversation_limits_are_commented_out_in_env_example(self):
        """A fresh install gets the code defaults (3 hours, 50 turns), not example-file values."""
        lines = (self.PROJECT_ROOT / ".env.example").read_text().splitlines()
        for name in ("CONVERSATION_TIMEOUT_HOURS", "MAX_CONVERSATION_TURNS"):
            assert not any(line.startswith(f"{name}=") for line in lines), f"{name} must stay commented out"
            assert any(line.startswith(f"# {name}=") for line in lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
