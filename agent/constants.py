"""
Central repository for codebase analysis constants.
Moving these here keeps business logic clean and allows for easy tuning.
"""

# Priorities for files (0-100). Higher means "included first".
SOURCE_EXTENSIONS: dict[str, int] = {
    ".py": 100,
    ".js": 95,
    ".ts": 95,
    ".tsx": 95,
    ".jsx": 95,
    ".mjs": 90,
    ".cjs": 90,
    ".json": 40,
    ".yaml": 35,
    ".yml": 35,
    ".toml": 35,
    ".md": 20,
    ".txt": 10,
}

# Files that are technically "text" but usually noise for an LLM
LOW_VALUE_FILENAMES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    ".ds_store",
    "tags",
    ".gitignore",
    ".dockerignore",
}

# Directories that should be skipped entirely
LOW_VALUE_DIRS: set[str] = {
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "target",  # Rust builds
}

# Keywords that boost a file's importance based on the user's prompt
TEST_HINTS: set[str] = {"test", "tests", "spec", "__tests__"}
CONFIG_HINTS: set[str] = {"config", "settings", "setup", "env"}

# Common English words to ignore when extracting keywords from the prompt
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "please",
    "add",
    "make",
    "change",
    "create",
    "fix",
    "update",
}
