"""
Configuration management for Healthcare AI Research Agent
"""

import os
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class CacheType(str, Enum):
    """Cache type enumeration"""
    IN_MEMORY = "IN_MEMORY"
    SQLITE = "SQLITE"

class ModelConfig(BaseModel):
    """Model configuration"""
    name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 50000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v

    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v < 100:
            raise ValueError('Max tokens must be at least 100')
        return v

class CacheConfig(BaseModel):
    """Cache configuration"""
    cache_type: CacheType = CacheType.IN_MEMORY
    memory_cache_size: int = 100
    sqlite_cache_path: str = "./cache/healthcare_cache.db"

    @validator('memory_cache_size')
    def validate_memory_cache_size(cls, v):
        if v < 1:
            raise ValueError('Memory cache size must be at least 1')
        return v

class BudgetConfig(BaseModel):
    """Budget and limit configuration"""
    max_iterations: int = 20
    max_tool_calls: int = 50
    max_tokens_per_run: int = 50000
    max_repair_rounds: int = 2

    @validator('max_iterations', 'max_tool_calls', 'max_repair_rounds')
    def validate_counts(cls, v):
        if v < 1:
            raise ValueError('Count values must be at least 1')
        return v

class ToolConfig(BaseModel):
    """Tool configuration"""
    enabled_tools: List[str] = [
        "search_information",
        "retrieve_document",
        "calculate_metric",
        "compare_companies",
        "save_research",
        "generate_report"
    ]
    search_max_results: int = 10
    search_recency_window: str = "1_year"
    tool_timeout_seconds: int = 30

    @validator('search_max_results')
    def validate_search_max_results(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Search max results must be between 1 and 50')
        return v

class Settings(BaseSettings):
    """
    Main configuration class for Healthcare AI Research Agent

    Loads configuration from environment variables and provides
    type-safe access to all application settings.
    """

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Cache configuration
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Budget configuration
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    # Tool configuration
    tools: ToolConfig = Field(default_factory=ToolConfig)

    # OpenAI API key (required)
    openai_api_key: Optional[str] = None

    # Application settings
    app_name: str = "Healthcare AI Research Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Direct access to common settings for backward compatibility
    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str):
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.model.temperature = value

    @property
    def max_tokens(self) -> int:
        return self.model.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self.model.max_tokens = value

    @property
    def cache_type(self) -> CacheType:
        return self.cache.cache_type

    @cache_type.setter
    def cache_type(self, value: CacheType):
        self.cache.cache_type = value

    @property
    def memory_cache_size(self) -> int:
        return self.cache.memory_cache_size

    @property
    def sqlite_cache_path(self) -> str:
        return self.cache.sqlite_cache_path

    @property
    def max_iterations(self) -> int:
        return self.budget.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int):
        self.budget.max_iterations = value

    @property
    def max_tool_calls(self) -> int:
        return self.budget.max_tool_calls

    @property
    def max_repair_rounds(self) -> int:
        return self.budget.max_repair_rounds

    # Configuration file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        """Initialize settings with environment variables"""
        super().__init__(**kwargs)

        # Validate OpenAI API key
        if not self.openai_api_key:
            # Try to get from environment
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            print("⚠️  WARNING: OpenAI API key not configured. Some features may not work.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "model": self.model.model_dump(),
            "cache": self.cache.model_dump(),
            "budget": self.budget.model_dump(),
            "tools": self.tools.model_dump(),
            "openai_api_key": "***REDACTED***" if self.openai_api_key else None,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "log_level": self.log_level
        }

    def save_to_env(self, path: str = ".env"):
        """Save current settings to .env file (excluding sensitive data)"""
        env_content = f"""
# {self.app_name} Configuration
# Generated automatically - edit with care

# Model Settings
MODEL_NAME={self.model.name}
TEMPERATURE={self.model.temperature}
MAX_TOKENS={self.model.max_tokens}

# Cache Settings
CACHE_TYPE={self.cache.cache_type.value}
MEMORY_CACHE_SIZE={self.cache.memory_cache_size}
SQLITE_CACHE_PATH={self.cache.sqlite_cache_path}

# Budget Settings
MAX_ITERATIONS={self.budget.max_iterations}
MAX_TOOL_CALLS={self.budget.max_tool_calls}
MAX_TOKENS_PER_RUN={self.budget.max_tokens_per_run}
MAX_REPAIR_ROUNDS={self.budget.max_repair_rounds}

# Tool Settings
SEARCH_MAX_RESULTS={self.tools.search_max_results}
SEARCH_RECENCY_WINDOW={self.tools.search_recency_window}
TOOL_TIMEOUT_SECONDS={self.tools.tool_timeout_seconds}

# Application Settings
APP_NAME={self.app_name}
APP_VERSION={self.app_version}
DEBUG={self.debug}
LOG_LEVEL={self.log_level}
"""

        with open(path, "w") as f:
            f.write(env_content.strip())

    @classmethod
    def from_env(cls, env_path: str = ".env") -> 'Settings':
        """Load settings from .env file"""
        return cls(_env_file=env_path)

# Create a global settings instance
settings = Settings()