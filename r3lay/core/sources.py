"""Source type classification and trust levels for RAG attribution.

This module provides source classification for the hybrid RAG system to:
1. Assign trust levels based on source type and origin
2. Generate appropriate citation prefixes for responses
3. Distinguish between local indexed content and web sources

Trust hierarchy (highest to lowest):
- INDEXED_CURATED: User-curated content - manuals, datasheets, saved pages, specs (1.0)
- INDEXED_IMAGE: Visual documentation, diagrams, PDF pages (1.0)
- INDEXED_DOCUMENT: General indexed documents (0.95)
- INDEXED_CODE: Source code from project (0.9)
- WEB_OE_FIRSTPARTY: Manufacturer official sites (0.85)
- WEB_TRUSTED: Reputable third-party sources (0.7)
- WEB_GENERAL: Other web sources (0.5)
- WEB_COMMUNITY: Forums, reddit, user-generated content (0.4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse


class SourceType(str, Enum):
    """Source types with trust hierarchy.

    Each source type has an associated trust level (0.0-1.0) and
    citation prefix for generating attributions in responses.
    """

    # Local indexed sources (highest trust)
    INDEXED_CURATED = "indexed_curated"  # User-curated: manuals, datasheets, saved pages, specs
    INDEXED_DOCUMENT = "indexed_document"  # General docs
    INDEXED_IMAGE = "indexed_image"  # Images, PDF pages
    INDEXED_CODE = "indexed_code"  # Source code

    # Web sources (varying trust)
    WEB_OE_FIRSTPARTY = "web_oe"  # Manufacturer/official sites
    WEB_TRUSTED = "web_trusted"  # Reputable third-party
    WEB_COMMUNITY = "web_community"  # Forums, reddit, etc.
    WEB_GENERAL = "web_general"  # Other web

    @property
    def trust_level(self) -> float:
        """Return trust level from 0.0 to 1.0.

        Trust levels determine how much weight to give information
        from this source and how it should be presented to users.
        """
        levels: dict[SourceType, float] = {
            SourceType.INDEXED_CURATED: 1.0,
            SourceType.INDEXED_IMAGE: 1.0,
            SourceType.INDEXED_DOCUMENT: 0.95,
            SourceType.INDEXED_CODE: 0.9,
            SourceType.WEB_OE_FIRSTPARTY: 0.85,
            SourceType.WEB_TRUSTED: 0.7,
            SourceType.WEB_GENERAL: 0.5,
            SourceType.WEB_COMMUNITY: 0.4,
        }
        return levels.get(self, 0.5)

    @property
    def citation_prefix(self) -> str:
        """Return template for introducing citations from this source.

        The template may contain placeholders:
        - {context}: The specific context/topic being discussed
        - {manufacturer}: The manufacturer name (for OE sources)
        - {source}: The source name/title
        """
        prefixes: dict[SourceType, str] = {
            SourceType.INDEXED_CURATED: "According to {context} documentation",
            SourceType.INDEXED_IMAGE: "From {context} documentation",
            SourceType.INDEXED_DOCUMENT: "From indexed documentation",
            SourceType.INDEXED_CODE: "From the codebase",
            SourceType.WEB_OE_FIRSTPARTY: "Per official {manufacturer} documentation",
            SourceType.WEB_TRUSTED: "According to {source}",
            SourceType.WEB_GENERAL: "Based on web search",
            SourceType.WEB_COMMUNITY: "Community discussion suggests (verify independently)",
        }
        return prefixes.get(self, "Based on available information")

    @property
    def is_local(self) -> bool:
        """Check if this is a local indexed source."""
        return self in {
            SourceType.INDEXED_CURATED,
            SourceType.INDEXED_DOCUMENT,
            SourceType.INDEXED_IMAGE,
            SourceType.INDEXED_CODE,
        }

    @property
    def is_web(self) -> bool:
        """Check if this is a web source."""
        return self in {
            SourceType.WEB_OE_FIRSTPARTY,
            SourceType.WEB_TRUSTED,
            SourceType.WEB_COMMUNITY,
            SourceType.WEB_GENERAL,
        }

    @property
    def requires_verification(self) -> bool:
        """Check if information from this source should be verified.

        Returns True for community sources and general web content
        where information quality is variable.
        """
        return self in {SourceType.WEB_COMMUNITY, SourceType.WEB_GENERAL}


# ============================================================================
# Domain Classification Lists
# ============================================================================

# OE/First-party domain patterns (automotive focus)
OE_DOMAINS: set[str] = {
    # Major automotive manufacturers
    "subaru.com",
    "toyota.com",
    "honda.com",
    "ford.com",
    "gm.com",
    "chevrolet.com",
    "nissan.com",
    "mazda.com",
    "hyundai.com",
    "kia.com",
    "bmw.com",
    "mercedes-benz.com",
    "audi.com",
    "vw.com",
    "volkswagen.com",
    "porsche.com",
    "lexus.com",
    "acura.com",
    "infiniti.com",
    "mitsubishi-motors.com",
    "tesla.com",
    "rivian.com",
    # Official parts/service sites
    "subaruparts.com",
    "parts.subaru.com",
    "hondapartsnow.com",
    "parts.honda.com",
    "toyotapartsdeal.com",
    "parts.ford.com",
    # Tech documentation sites
    "developer.apple.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "docs.python.org",
    "docs.rust-lang.org",
    "docs.docker.com",
    "kubernetes.io",
}

# Trusted third-party domains
TRUSTED_DOMAINS: set[str] = {
    # Professional automotive repair databases
    "alldata.com",
    "mitchellondemand.com",
    "identifix.com",
    "shopkey.com",
    # Parts suppliers with reliable specs
    "rockauto.com",
    "partsgeek.com",
    "fcpeuro.com",
    "ecstuning.com",
    # Established automotive media
    "caranddriver.com",
    "motortrend.com",
    "edmunds.com",
    "kbb.com",
    "autoblog.com",
    "roadandtrack.com",
    # Technical reference
    "wikipedia.org",
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    # Documentation sites
    "readthedocs.io",
    "readthedocs.org",
    "rtfd.io",
    "pypi.org",
    "npmjs.com",
    "crates.io",
}

# Community/forum domains (lower trust - user-generated content)
COMMUNITY_DOMAINS: set[str] = {
    # General forums
    "reddit.com",
    "quora.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    # Tech Q&A
    "stackoverflow.com",
    "stackexchange.com",
    "superuser.com",
    "serverfault.com",
    # Automotive enthusiast forums
    "forums.nasioc.com",
    "clubwrx.net",
    "subaruforester.org",
    "ft86club.com",
    "civicx.com",
    "focusst.org",
    "focusrs.org",
    "mustang6g.com",
    "corvetteforum.com",
    "bimmerpost.com",
    "audizine.com",
    "vwvortex.com",
    "miataforum.com",
}

# Patterns that indicate community/forum content
COMMUNITY_PATTERNS: list[str] = [
    r"forum",
    r"community",
    r"discuss",
    r"reddit\.com",
    r"\.club$",
    r"enthusiast",
    r"owner[s]?\.com",
    r"talk\.",
    r"boards\.",
]


# ============================================================================
# Path Patterns for Local Files
# ============================================================================

# File extensions for code
CODE_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".sql",
    ".r",
    ".R",
    ".m",
    ".mm",
}

# File extensions for images
IMAGE_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
    ".heic",
    ".heif",
}

# Keywords that indicate curated documentation content
# These are matched as whole words (with path-aware boundaries) to avoid
# false positives like "inspect" matching "spec".
#
# Note: "reference" was removed as it's too common in code paths
# (e.g., "rules_block/reference.py"). Use more specific keywords instead.
CURATED_KEYWORDS: set[str] = {
    "fsm",
    "manual",
    "manuals",
    "service",
    "repair",
    "workshop",
    "factory",
    "technical",
    "specification",
    "specifications",
    "spec",
    "specs",
    "bulletin",
    "tsb",
    "procedure",
    "procedures",
    "maintenance",
    "diagnostic",
    "diagnostics",
    "datasheet",
    "datasheets",
    "guide",
    "guides",
    "handbook",
}

# Compiled regex pattern for path-aware word boundary matching of curated keywords.
# Uses lookahead/lookbehind for path separators (/, \, _, -) as word boundaries,
# in addition to standard word boundaries. This allows matching:
#   - "docs/service_manual.pdf" -> matches "service" and "manual"
#   - "2020-outback-specs.pdf" -> matches "specs"
# But NOT:
#   - "inspect.json" -> should not match "spec"
_CURATED_PATTERN = re.compile(
    r"(?:^|[/_\-\s])(" + "|".join(re.escape(kw) for kw in CURATED_KEYWORDS) + r")(?:[/_\-\s.]|$)",
    re.IGNORECASE,
)


# ============================================================================
# Detection Functions
# ============================================================================


def detect_source_type_from_path(path: Path) -> SourceType:
    """Detect source type from local file path.

    Analyzes the file extension and path components to determine
    the appropriate source type classification.

    Uses word boundary matching for curated keywords to avoid false
    positives (e.g., "inspect" should not match "spec").

    Args:
        path: Path to the local file.

    Returns:
        SourceType based on file characteristics.

    Examples:
        >>> detect_source_type_from_path(Path("docs/fsm/engine.pdf"))
        SourceType.INDEXED_CURATED
        >>> detect_source_type_from_path(Path("src/main.py"))
        SourceType.INDEXED_CODE
        >>> detect_source_type_from_path(Path("images/diagram.png"))
        SourceType.INDEXED_IMAGE
        >>> detect_source_type_from_path(Path("cache/inspect.json"))
        SourceType.INDEXED_DOCUMENT  # "inspect" should NOT match "spec"
    """
    path_str = str(path).lower()
    suffix = path.suffix.lower()

    # Images
    if suffix in IMAGE_EXTENSIONS:
        return SourceType.INDEXED_IMAGE

    # PDFs - check if curated content based on path (word boundary match)
    if suffix == ".pdf":
        if _CURATED_PATTERN.search(path_str):
            return SourceType.INDEXED_CURATED
        return SourceType.INDEXED_IMAGE  # PDF pages treated as images

    # Code files
    if suffix in CODE_EXTENSIONS:
        return SourceType.INDEXED_CODE

    # Curated content detection by path keywords (word boundary match)
    if _CURATED_PATTERN.search(path_str):
        return SourceType.INDEXED_CURATED

    # Default to general document
    return SourceType.INDEXED_DOCUMENT


def detect_source_type_from_url(url: str) -> SourceType:
    """Detect source type from web URL.

    Analyzes the domain and URL patterns to determine the appropriate
    source type classification for web content.

    Args:
        url: Full URL string.

    Returns:
        SourceType based on URL characteristics.

    Examples:
        >>> detect_source_type_from_url("https://subaru.com/service/manual")
        SourceType.WEB_OE_FIRSTPARTY
        >>> detect_source_type_from_url("https://reddit.com/r/subaru")
        SourceType.WEB_COMMUNITY
        >>> detect_source_type_from_url("https://alldata.com/repair/engine")
        SourceType.WEB_TRUSTED
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
    except Exception:
        return SourceType.WEB_GENERAL

    # Check OE/first-party domains
    if domain in OE_DOMAINS:
        return SourceType.WEB_OE_FIRSTPARTY
    # Also check if domain ends with an OE domain (subdomains)
    for oe_domain in OE_DOMAINS:
        if domain.endswith("." + oe_domain) or domain == oe_domain:
            return SourceType.WEB_OE_FIRSTPARTY

    # Check trusted third-party domains
    if domain in TRUSTED_DOMAINS:
        return SourceType.WEB_TRUSTED
    for trusted_domain in TRUSTED_DOMAINS:
        if domain.endswith("." + trusted_domain) or domain == trusted_domain:
            return SourceType.WEB_TRUSTED

    # Check community domains
    if domain in COMMUNITY_DOMAINS:
        return SourceType.WEB_COMMUNITY
    for community_domain in COMMUNITY_DOMAINS:
        if domain.endswith("." + community_domain) or domain == community_domain:
            return SourceType.WEB_COMMUNITY

    # Check community URL patterns
    url_lower = url.lower()
    for pattern in COMMUNITY_PATTERNS:
        if re.search(pattern, url_lower):
            return SourceType.WEB_COMMUNITY

    # Default to general web
    return SourceType.WEB_GENERAL


def format_citation(
    source_type: SourceType,
    context: str | None = None,
    manufacturer: str | None = None,
    source: str | None = None,
) -> str:
    """Format a citation prefix with the given context values.

    Args:
        source_type: The type of source being cited.
        context: Context/topic for the citation (e.g., "2020 Outback").
        manufacturer: Manufacturer name for OE sources.
        source: Source name/title for trusted sources.

    Returns:
        Formatted citation prefix string.

    Examples:
        >>> format_citation(SourceType.INDEXED_CURATED, context="2020 Outback")
        "According to 2020 Outback documentation"
        >>> format_citation(SourceType.WEB_OE_FIRSTPARTY, manufacturer="Subaru")
        "Per official Subaru documentation"
    """
    template = source_type.citation_prefix

    # Build replacement dict
    replacements = {
        "context": context or "the",
        "manufacturer": manufacturer or "manufacturer",
        "source": source or "external sources",
    }

    # Apply replacements
    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)

    return result


@dataclass
class SourceInfo:
    """Complete source information for a retrieved chunk.

    Combines source type with additional metadata for rich attribution.
    """

    source_type: SourceType
    path: str | None = None  # Local path or URL
    title: str | None = None  # Document/page title
    section: str | None = None  # Section within document
    page: int | None = None  # Page number (for PDFs)
    manufacturer: str | None = None  # Extracted manufacturer name

    @property
    def trust_level(self) -> float:
        """Get trust level from source type."""
        return self.source_type.trust_level

    @property
    def is_local(self) -> bool:
        """Check if this is a local indexed source."""
        return self.source_type.is_local

    def format_citation(self, context: str | None = None) -> str:
        """Format a citation for this source.

        Args:
            context: Optional context to include in citation.

        Returns:
            Formatted citation string.
        """
        return format_citation(
            source_type=self.source_type,
            context=context or self.title,
            manufacturer=self.manufacturer,
            source=self.title,
        )


__all__ = [
    # Main enum
    "SourceType",
    # Detection functions
    "detect_source_type_from_path",
    "detect_source_type_from_url",
    "format_citation",
    # Data classes
    "SourceInfo",
    # Domain lists (for customization)
    "OE_DOMAINS",
    "TRUSTED_DOMAINS",
    "COMMUNITY_DOMAINS",
    "COMMUNITY_PATTERNS",
    # Extension lists
    "CODE_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "CURATED_KEYWORDS",
]
