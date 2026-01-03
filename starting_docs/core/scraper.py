"""Pipet-style web scraper using YAML selector configs."""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from ruamel.yaml import YAML


@dataclass
class ScraperConfig:
    """Configuration for a scraper."""
    name: str
    url_pattern: str
    selectors: dict[str, str]
    transform: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def matches_url(self, url: str) -> bool:
        """Check if this scraper config matches a URL."""
        pattern = self.url_pattern.replace("*", ".*")
        return bool(re.match(pattern, url))


@dataclass
class ScrapedContent:
    """Content extracted from a page."""
    url: str
    data: dict[str, Any]
    raw_html: str | None = None
    scraper_name: str | None = None


class PipetScraper:
    """
    Pipet-style scraper using YAML selector definitions.
    
    Uses external pipet binary if available, falls back to BeautifulSoup.
    """
    
    def __init__(self, scrapers_path: Path | None = None):
        self.scrapers_path = scrapers_path
        self.configs: dict[str, ScraperConfig] = {}
        self._has_pipet = self._check_pipet()
        
        if scrapers_path and scrapers_path.exists():
            self._load_scrapers()
    
    def _check_pipet(self) -> bool:
        """Check if pipet binary is available."""
        try:
            result = subprocess.run(["pipet", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _load_scrapers(self) -> None:
        """Load scraper configs from YAML files."""
        yaml = YAML()
        for config_file in self.scrapers_path.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    data = yaml.load(f)
                    if data:
                        config = ScraperConfig(
                            name=config_file.stem,
                            url_pattern=data.get("url_pattern", ""),
                            selectors=data.get("selectors", {}),
                            transform=data.get("transform"),
                            metadata=data.get("metadata", {}),
                        )
                        self.configs[config.name] = config
            except Exception as e:
                print(f"Failed to load scraper {config_file}: {e}")
    
    def add_config(self, config: ScraperConfig) -> None:
        """Add a scraper configuration."""
        self.configs[config.name] = config
    
    def find_config(self, url: str) -> ScraperConfig | None:
        """Find a matching scraper config for a URL."""
        for config in self.configs.values():
            if config.matches_url(url):
                return config
        return None
    
    async def scrape(
        self,
        url: str,
        config: ScraperConfig | None = None,
    ) -> ScrapedContent:
        """Scrape a URL using the appropriate config."""
        if config is None:
            config = self.find_config(url)
        
        if config is None:
            return await self._scrape_basic(url)
        
        if self._has_pipet:
            return await self._scrape_pipet(url, config)
        else:
            return await self._scrape_bs4(url, config)
    
    async def _scrape_basic(self, url: str) -> ScrapedContent:
        """Basic scraping without specific selectors."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        
        # Extract text content
        from html.parser import HTMLParser
        
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.title = ""
                self._in_title = False
                self._skip = False
            
            def handle_starttag(self, tag, attrs):
                if tag == "title":
                    self._in_title = True
                elif tag in ("script", "style", "head", "nav", "footer"):
                    self._skip = True
            
            def handle_endtag(self, tag):
                if tag == "title":
                    self._in_title = False
                elif tag in ("script", "style", "head", "nav", "footer"):
                    self._skip = False
            
            def handle_data(self, data):
                if self._in_title:
                    self.title = data.strip()
                elif not self._skip:
                    text = data.strip()
                    if text:
                        self.text_parts.append(text)
        
        parser = TextExtractor()
        parser.feed(html)
        
        return ScrapedContent(
            url=url,
            data={
                "title": parser.title,
                "content": " ".join(parser.text_parts),
            },
            raw_html=html,
        )
    
    async def _scrape_pipet(self, url: str, config: ScraperConfig) -> ScrapedContent:
        """Scrape using external pipet binary."""
        import json
        import tempfile
        
        yaml = YAML()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "url_pattern": config.url_pattern,
                "selectors": config.selectors,
            }, f)
            config_path = f.name
        
        try:
            result = subprocess.run(
                ["pipet", config_path, url],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise ScraperError(f"Pipet failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            return ScrapedContent(
                url=url,
                data=data,
                scraper_name=config.name,
            )
        
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    async def _scrape_bs4(self, url: str, config: ScraperConfig) -> ScrapedContent:
        """Scrape using BeautifulSoup."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            data = {}
            
            for key, selector in config.selectors.items():
                elements = soup.select(selector)
                if len(elements) == 1:
                    data[key] = elements[0].get_text(strip=True)
                elif len(elements) > 1:
                    data[key] = [el.get_text(strip=True) for el in elements]
                else:
                    data[key] = None
            
            return ScrapedContent(
                url=url,
                data=data,
                raw_html=html,
                scraper_name=config.name,
            )
        
        except ImportError:
            return await self._scrape_basic(url)
    
    def save_config(self, config: ScraperConfig) -> Path:
        """Save a scraper config to disk."""
        if not self.scrapers_path:
            raise ValueError("No scrapers path configured")
        
        self.scrapers_path.mkdir(parents=True, exist_ok=True)
        config_path = self.scrapers_path / f"{config.name}.yaml"
        
        yaml = YAML()
        yaml.default_flow_style = False
        
        with open(config_path, "w") as f:
            yaml.dump({
                "url_pattern": config.url_pattern,
                "selectors": config.selectors,
                "transform": config.transform,
                "metadata": config.metadata,
            }, f)
        
        self.configs[config.name] = config
        return config_path


class ScraperError(Exception):
    """Scraping operation failed."""
    pass
