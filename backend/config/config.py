"""
Application configuration with environment-based settings.
All configuration is explicit, validated, and logged at startup.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for development but require
    explicit configuration in production environments.
    """
    
    model_config = SettingsConfigDict(
        env_file="../.env",  # Load from project root (relative to backend/)
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )
    
    # Application
    app_name: str = Field(default="Qualified Health", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    
    # LLM Provider (DeepSeek)
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API base URL")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    llm_max_tokens: int = Field(default=2048, description="Max tokens per response")
    llm_temperature: float = Field(default=1.3, description="Model temperature")
    
    # OpenAI (for embeddings)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings (set OPENAI_API_KEY in .env)",
    )
    
    # ArangoDB - Using ARANGODB_USERNAME and ARANGODB_PASSWORD from environment
    # Note: This is for direct ArangoDB connections (separate from GraphRAG endpoint)
    arango_host: str = Field(
        default="https://arangodb-platform-dev.pilot.arangodb.com:8529",
        description="ArangoDB host URL for direct database connections"
    )
    # Field names match env var names for pydantic-settings
    ARANGODB_USERNAME: str = Field(
        default="root",
        description="ArangoDB username (from ARANGODB_USERNAME env var)"
    )
    ARANGODB_PASSWORD: str = Field(
        default="",
        description="ArangoDB password (from ARANGODB_PASSWORD env var)"
    )
    
    # Properties for backward compatibility with existing code
    @property
    def arango_username(self) -> str:
        """Get ArangoDB username (maps from ARANGODB_USERNAME env var)."""
        return self.ARANGODB_USERNAME
    
    @property
    def arango_password(self) -> str:
        """Get ArangoDB password (maps from ARANGODB_PASSWORD env var)."""
        return self.ARANGODB_PASSWORD
    arango_database: str = Field(default="ary_db", description="ArangoDB database name")
    
    # GraphRAG - Hardcoded values matching notebook
    # Notebook uses: SERVER_URL = os.environ['ARANGO_DEPLOYMENT_ENDPOINT']
    # Using external URL for local development (not Kubernetes .svc internal URL)
    arango_deployment_endpoint: str = Field(
        default="https://arangodb-platform-dev.pilot.arangodb.com:8529",
        description="ArangoDB deployment endpoint (base URL) - hardcoded external URL for local dev"
    )
    graphrag_service_id: str = Field(
        default="dcajr",
        description="GraphRAG retriever service ID - hardcoded"
    )
    graphrag_project_name: str = Field(
        default="test",
        description="GraphRAG project name - hardcoded"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format"
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def get_safe_config_dict(self) -> dict:
        """Return configuration dict with secrets redacted for logging."""
        config = self.model_dump()
        # Redact sensitive values
        if config.get("deepseek_api_key"):
            config["deepseek_api_key"] = "***REDACTED***"
        if config.get("openai_api_key"):
            config["openai_api_key"] = "***REDACTED***"
        if config.get("ARANGODB_PASSWORD"):
            config["ARANGODB_PASSWORD"] = "***REDACTED***"
        if config.get("arango_password"):
            config["arango_password"] = "***REDACTED***"
        return config


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Settings are loaded once and cached for the application lifetime.
    Use dependency injection in FastAPI routes for testability.
    """
    return Settings()
