"""Tests for the source classification and trust level module.

Covers:
- SourceType enum values and properties
- detect_source_type_from_path function
- detect_source_type_from_url function
- format_citation function
- SourceInfo dataclass
"""

from __future__ import annotations

from pathlib import Path

import pytest

from r3lay.core.sources import (
    SourceType,
    SourceInfo,
    detect_source_type_from_path,
    detect_source_type_from_url,
    format_citation,
    OE_DOMAINS,
    TRUSTED_DOMAINS,
    COMMUNITY_DOMAINS,
    CODE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    CURATED_KEYWORDS,
)


# ============================================================================
# SourceType Enum Tests
# ============================================================================


class TestSourceType:
    """Tests for SourceType enum."""

    def test_all_types_exist(self):
        """All expected source types are defined."""
        expected = [
            "INDEXED_CURATED",
            "INDEXED_DOCUMENT",
            "INDEXED_IMAGE",
            "INDEXED_CODE",
            "WEB_OE_FIRSTPARTY",
            "WEB_TRUSTED",
            "WEB_COMMUNITY",
            "WEB_GENERAL",
        ]
        for name in expected:
            assert hasattr(SourceType, name)

    def test_trust_level_range(self):
        """All trust levels are between 0.0 and 1.0."""
        for source_type in SourceType:
            assert 0.0 <= source_type.trust_level <= 1.0

    def test_trust_level_hierarchy(self):
        """Trust levels follow expected hierarchy."""
        # Curated local content should have highest trust
        assert SourceType.INDEXED_CURATED.trust_level == 1.0
        assert SourceType.INDEXED_IMAGE.trust_level == 1.0

        # OE should be higher than general web
        assert SourceType.WEB_OE_FIRSTPARTY.trust_level > SourceType.WEB_GENERAL.trust_level

        # Community content should have lowest web trust
        assert SourceType.WEB_COMMUNITY.trust_level < SourceType.WEB_TRUSTED.trust_level

    def test_citation_prefix_not_empty(self):
        """All source types have citation prefixes."""
        for source_type in SourceType:
            assert source_type.citation_prefix
            assert len(source_type.citation_prefix) > 0

    def test_is_local_property(self):
        """is_local correctly identifies local sources."""
        local_types = [
            SourceType.INDEXED_CURATED,
            SourceType.INDEXED_DOCUMENT,
            SourceType.INDEXED_IMAGE,
            SourceType.INDEXED_CODE,
        ]
        web_types = [
            SourceType.WEB_OE_FIRSTPARTY,
            SourceType.WEB_TRUSTED,
            SourceType.WEB_COMMUNITY,
            SourceType.WEB_GENERAL,
        ]

        for st in local_types:
            assert st.is_local is True
        for st in web_types:
            assert st.is_local is False

    def test_is_web_property(self):
        """is_web correctly identifies web sources."""
        local_types = [
            SourceType.INDEXED_CURATED,
            SourceType.INDEXED_DOCUMENT,
            SourceType.INDEXED_IMAGE,
            SourceType.INDEXED_CODE,
        ]
        web_types = [
            SourceType.WEB_OE_FIRSTPARTY,
            SourceType.WEB_TRUSTED,
            SourceType.WEB_COMMUNITY,
            SourceType.WEB_GENERAL,
        ]

        for st in local_types:
            assert st.is_web is False
        for st in web_types:
            assert st.is_web is True

    def test_requires_verification(self):
        """requires_verification identifies low-trust sources."""
        assert SourceType.WEB_COMMUNITY.requires_verification is True
        assert SourceType.WEB_GENERAL.requires_verification is True
        assert SourceType.INDEXED_CURATED.requires_verification is False
        assert SourceType.WEB_OE_FIRSTPARTY.requires_verification is False


# ============================================================================
# detect_source_type_from_path Tests
# ============================================================================


class TestDetectSourceTypeFromPath:
    """Tests for detect_source_type_from_path function."""

    def test_code_files(self):
        """Code files are classified as INDEXED_CODE."""
        code_paths = [
            Path("src/main.py"),
            Path("app/index.ts"),
            Path("lib/utils.js"),
            Path("pkg/server.go"),
            Path("core/lib.rs"),
        ]
        for path in code_paths:
            assert detect_source_type_from_path(path) == SourceType.INDEXED_CODE

    def test_image_files(self):
        """Image files are classified as INDEXED_IMAGE."""
        image_paths = [
            Path("images/diagram.png"),
            Path("assets/photo.jpg"),
            Path("docs/screenshot.jpeg"),
            Path("media/animation.gif"),
        ]
        for path in image_paths:
            assert detect_source_type_from_path(path) == SourceType.INDEXED_IMAGE

    def test_curated_pdf(self):
        """PDFs with curated keywords are INDEXED_CURATED."""
        curated_paths = [
            Path("docs/fsm/engine.pdf"),
            Path("manuals/2020-outback-service-manual.pdf"),
            Path("specs/datasheet.pdf"),
            Path("docs/workshop/procedures.pdf"),
        ]
        for path in curated_paths:
            assert detect_source_type_from_path(path) == SourceType.INDEXED_CURATED

    def test_generic_pdf(self):
        """PDFs without curated keywords are INDEXED_IMAGE."""
        generic_paths = [
            Path("downloads/document.pdf"),
            Path("cache/file.pdf"),
            Path("temp/output.pdf"),
        ]
        for path in generic_paths:
            assert detect_source_type_from_path(path) == SourceType.INDEXED_IMAGE

    def test_curated_keyword_word_boundary(self):
        """Curated keywords must match word boundaries."""
        # "inspect" should NOT match "spec"
        path = Path("cache/inspect.json")
        result = detect_source_type_from_path(path)
        assert result == SourceType.INDEXED_DOCUMENT  # Not CURATED

    def test_curated_non_pdf_by_path(self):
        """Non-PDF files with curated keywords are INDEXED_CURATED."""
        path = Path("docs/service_manual/chapter1.md")
        assert detect_source_type_from_path(path) == SourceType.INDEXED_CURATED

    def test_generic_document(self):
        """Generic documents default to INDEXED_DOCUMENT."""
        paths = [
            Path("notes/readme.txt"),
            Path("data/config.yaml"),
            Path("cache/data.json"),
        ]
        for path in paths:
            assert detect_source_type_from_path(path) == SourceType.INDEXED_DOCUMENT


# ============================================================================
# detect_source_type_from_url Tests
# ============================================================================


class TestDetectSourceTypeFromUrl:
    """Tests for detect_source_type_from_url function."""

    def test_oe_domains(self):
        """OE domains are classified as WEB_OE_FIRSTPARTY."""
        oe_urls = [
            "https://subaru.com/service/manual",
            "https://www.toyota.com/owners",
            "https://developer.apple.com/documentation",
            "https://docs.python.org/3/library/",
        ]
        for url in oe_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_OE_FIRSTPARTY

    def test_oe_subdomains(self):
        """Subdomains of OE domains are also WEB_OE_FIRSTPARTY."""
        subdomain_urls = [
            "https://parts.subaru.com/catalog",
            "https://service.honda.com/repair",
        ]
        for url in subdomain_urls:
            result = detect_source_type_from_url(url)
            assert result == SourceType.WEB_OE_FIRSTPARTY

    def test_trusted_domains(self):
        """Trusted domains are classified as WEB_TRUSTED."""
        trusted_urls = [
            "https://rockauto.com/parts/search",
            "https://github.com/user/repo",
            "https://en.wikipedia.org/wiki/Topic",
            "https://readthedocs.io/projects/myproject",
        ]
        for url in trusted_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_TRUSTED

    def test_community_domains(self):
        """Community domains are classified as WEB_COMMUNITY."""
        community_urls = [
            "https://reddit.com/r/subaru",
            "https://stackoverflow.com/questions/123",
            "https://forums.nasioc.com/forums/showthread.php?t=123",
            "https://www.facebook.com/groups/subaruclub",
        ]
        for url in community_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_COMMUNITY

    def test_community_patterns(self):
        """URLs matching community patterns are WEB_COMMUNITY."""
        pattern_urls = [
            "https://somesite.com/forum/thread/123",
            "https://car-enthusiast-club.com/discussion",
            "https://owners.example.com/posts",
        ]
        for url in pattern_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_COMMUNITY

    def test_general_web(self):
        """Unknown domains default to WEB_GENERAL."""
        general_urls = [
            "https://random-blog.com/post/123",
            "https://news.example.org/article",
            "https://shop.somestore.net/product",
        ]
        for url in general_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_GENERAL

    def test_invalid_url(self):
        """Invalid URLs default to WEB_GENERAL."""
        invalid_urls = [
            "not-a-url",
            "",
            "ftp://invalid",
        ]
        for url in invalid_urls:
            assert detect_source_type_from_url(url) == SourceType.WEB_GENERAL


# ============================================================================
# format_citation Tests
# ============================================================================


class TestFormatCitation:
    """Tests for format_citation function."""

    def test_curated_with_context(self):
        """INDEXED_CURATED uses context in citation."""
        result = format_citation(SourceType.INDEXED_CURATED, context="2020 Outback")
        assert "2020 Outback" in result
        assert "documentation" in result.lower()

    def test_oe_with_manufacturer(self):
        """WEB_OE_FIRSTPARTY uses manufacturer name."""
        result = format_citation(SourceType.WEB_OE_FIRSTPARTY, manufacturer="Subaru")
        assert "Subaru" in result
        assert "official" in result.lower()

    def test_trusted_with_source(self):
        """WEB_TRUSTED uses source name."""
        result = format_citation(SourceType.WEB_TRUSTED, source="RockAuto")
        assert "RockAuto" in result

    def test_community_warning(self):
        """WEB_COMMUNITY includes verification warning."""
        result = format_citation(SourceType.WEB_COMMUNITY)
        assert "verify" in result.lower() or "community" in result.lower()

    def test_defaults_without_params(self):
        """Defaults are applied when params not provided."""
        result = format_citation(SourceType.INDEXED_CURATED)
        # Should still produce a valid string
        assert len(result) > 0
        assert "{" not in result  # No unreplaced placeholders


# ============================================================================
# SourceInfo Tests
# ============================================================================


class TestSourceInfo:
    """Tests for SourceInfo dataclass."""

    def test_basic_creation(self):
        """SourceInfo can be created with minimal args."""
        info = SourceInfo(source_type=SourceType.INDEXED_DOCUMENT)
        assert info.source_type == SourceType.INDEXED_DOCUMENT
        assert info.path is None
        assert info.title is None

    def test_full_creation(self):
        """SourceInfo can be created with all fields."""
        info = SourceInfo(
            source_type=SourceType.INDEXED_CURATED,
            path="/docs/manual.pdf",
            title="Service Manual",
            section="Chapter 3",
            page=42,
            manufacturer="Subaru",
        )
        assert info.path == "/docs/manual.pdf"
        assert info.title == "Service Manual"
        assert info.section == "Chapter 3"
        assert info.page == 42
        assert info.manufacturer == "Subaru"

    def test_trust_level_property(self):
        """trust_level delegates to source_type."""
        info = SourceInfo(source_type=SourceType.WEB_COMMUNITY)
        assert info.trust_level == SourceType.WEB_COMMUNITY.trust_level

    def test_is_local_property(self):
        """is_local delegates to source_type."""
        local_info = SourceInfo(source_type=SourceType.INDEXED_CODE)
        web_info = SourceInfo(source_type=SourceType.WEB_GENERAL)

        assert local_info.is_local is True
        assert web_info.is_local is False

    def test_format_citation_method(self):
        """format_citation uses title as context."""
        info = SourceInfo(
            source_type=SourceType.INDEXED_CURATED,
            title="2020 Outback FSM",
        )
        result = info.format_citation()
        assert "2020 Outback FSM" in result

    def test_format_citation_with_override(self):
        """format_citation can use explicit context."""
        info = SourceInfo(
            source_type=SourceType.INDEXED_CURATED,
            title="Some Title",
        )
        result = info.format_citation(context="Custom Context")
        assert "Custom Context" in result


# ============================================================================
# Domain/Extension Lists Tests
# ============================================================================


class TestDomainLists:
    """Tests for domain classification lists."""

    def test_oe_domains_not_empty(self):
        """OE_DOMAINS is populated."""
        assert len(OE_DOMAINS) > 0

    def test_trusted_domains_not_empty(self):
        """TRUSTED_DOMAINS is populated."""
        assert len(TRUSTED_DOMAINS) > 0

    def test_community_domains_not_empty(self):
        """COMMUNITY_DOMAINS is populated."""
        assert len(COMMUNITY_DOMAINS) > 0

    def test_no_overlap_oe_community(self):
        """OE and community domains don't overlap."""
        overlap = OE_DOMAINS & COMMUNITY_DOMAINS
        assert len(overlap) == 0

    def test_code_extensions_lowercase(self):
        """Code extensions are lowercase."""
        for ext in CODE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")

    def test_image_extensions_lowercase(self):
        """Image extensions are lowercase."""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")

    def test_curated_keywords_lowercase(self):
        """Curated keywords are lowercase."""
        for kw in CURATED_KEYWORDS:
            assert kw == kw.lower()
