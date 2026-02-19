# r3LAY Automotive Diagnostic Module

**Self-hosted automotive diagnostics knowledge base for garage hobbyists**

## What This Is

A curated collection of automotive diagnostic information optimized for r3LAY's hybrid RAG system. Covers OBD2/SSM1 diagnostics, common trouble codes, repair procedures, and decision flowcharts — all locally indexed for instant, private access.

**Target audience:** Garage hobbyists working on 1997-2001 Subaru vehicles (EJ22/EJ25 engines), but principles apply broadly to OBD2-era cars (1996+).

## What's Included

### Core Documentation
- **`obd2-codes-p0xxx.md`** - Generic OBD2 codes (all manufacturers)
  - Common P0XXX powertrain codes
  - Sensor diagnostics (MAF, O2, ECT, CPS)
  - Fuel/ignition/emissions troubleshooting
  - EJ22-specific notes and specs

- **`subaru-p1xxx-codes.md`** - Subaru manufacturer-specific codes
  - P1XXX Subaru extensions
  - Known issues by model year
  - IAC valve, knock sensor, ignition coil codes
  - SSM1 parameter monitoring

- **`ssm1-protocol-evoscan.md`** - Evoscan setup guide
  - SSM1 protocol overview
  - Hardware requirements ($20-50 total)
  - Datalogging for diagnostics
  - Real-time parameter monitoring

- **`diagnostic-flowcharts.md`** - Decision trees for common issues
  - Check engine light workflow
  - Misfire, fuel trim, no-start, rough idle flowcharts
  - DIY vs. shop decision matrix
  - Cost estimates and tool requirements

### Quick Reference
- **`quick-reference.md`** - At-a-glance tables
  - Common codes ranked by frequency
  - Tool cost breakdown
  - Sensor specs and test procedures
  - Maintenance intervals

## How to Use with r3LAY

### Option 1: Index All Automotive Docs (Recommended)
```bash
# Index the entire automotive directory
cd ~/repos/r3LAY
r3lay index docs/automotive/

# Now query naturally:
r3lay "P0171 code on my 97 Impreza"
r3lay "How to test MAF sensor with multimeter"
r3lay "Evoscan setup for EJ22"
```

r3LAY will:
1. Search indexed docs (INDEXED_CURATED = trust level 1.0)
2. Combine with web search if needed
3. Cite sources with attribution

### Option 2: Add to Project-Specific Index
```bash
# If tracking a specific vehicle project:
cd ~/vehicles/1997-impreza/
r3lay index ~/repos/r3LAY/docs/automotive/

# Queries will prioritize this documentation
r3lay "What's normal fuel pressure for EJ22?"
```

### Option 3: Quick Lookup Without Indexing
```bash
# Direct file reference (no indexing needed)
r3lay "Summarize ~/repos/r3LAY/docs/automotive/obd2-codes-p0xxx.md P0420"
```

## Example Queries

### Diagnostic Codes
```
r3lay "P0300 misfire code diagnosis"
→ Returns: Misfire flowchart + common causes + test procedures

r3lay "What does P1325 mean on Subaru?"
→ Returns: Knock sensor circuit (right bank), location, test procedure

r3lay "All EVAP codes explained"
→ Returns: P0440-P0457 codes + common fixes + cost estimates
```

### Repair Procedures
```
r3lay "How to clean MAF sensor"
→ Returns: Step-by-step, tools needed, cost

r3lay "Ignition coil testing procedure"
→ Returns: Multimeter test specs, swap test, replacement cost

r3lay "Best way to find vacuum leak"
→ Returns: Smoke test, carb cleaner spray test, common leak points
```

### Tool Setup
```
r3lay "Evoscan setup on Windows 11"
→ Returns: Cable requirements, software install, connection troubleshooting

r3lay "What tools do I need for basic diagnostics?"
→ Returns: Basic tool kit, cost breakdown, DIY vs. shop decisions
```

### Vehicle-Specific
```
r3lay "Common issues 1997 Impreza EJ22"
→ Returns: Crank sensor, knock sensor, EVAP, ranked by frequency

r3lay "EJ22 compression test specs"
→ Returns: Healthy range (145-190 psi), procedure, interpretation
```

## Trust Levels & Citations

When r3LAY uses this automotive knowledge:
- **Source Type:** INDEXED_CURATED (highest trust)
- **Trust Level:** 1.0 (user-curated documentation)
- **Citations:** "According to automotive documentation: ..."

This means:
- ✅ Prioritized over web search results
- ✅ Cited with high confidence
- ✅ Available offline (no web search needed for basics)

## Maintenance & Updates

### Adding New Content
```bash
# Add new markdown files to docs/automotive/
# Then re-index:
cd ~/repos/r3LAY
r3lay index docs/automotive/ --update
```

### Suggested Additions (Community-Driven)
- **More vehicle-specific modules** (Honda, Toyota, VW, etc.)
- **Advanced topics** (turbo tuning, standalone ECU)
- **Tool reviews** (cheap vs. expensive OBD2 scanners)
- **Real repair logs** (before/after diagnostics)

## Cost Analysis

### Self-Diagnosis Setup Cost
| Item | Cost | Notes |
|------|------|-------|
| Basic OBD2 scanner | $20-50 | Code reading only |
| Evoscan cable (SSM1) | $20-50 | Subaru-specific datalogging |
| Multimeter | $20-40 | Essential for testing |
| Compression tester | $30-50 | Optional but useful |
| **Total** | **$90-190** | One-time investment |

### Labor Cost Savings (Per Repair)
| Repair | Shop Cost | DIY Cost | Savings |
|--------|-----------|----------|---------|
| Read codes | $50-100 | $0 (own scanner) | $50-100 |
| Replace O2 sensor | $200-300 | $80-120 (part only) | $120-180 |
| Replace ignition coil | $150-250 | $60-80 | $90-170 |
| Clean MAF/TB | $100-150 | $10-20 (cleaner) | $80-130 |
| **Typical Annual** | **$500-800** | **$150-300** | **$350-500** |

**ROI:** Tool investment pays for itself after 1-2 repairs.

## Philosophy: Local-First Diagnostics

### Why This Matters
1. **Privacy** - Your vehicle data stays local (no cloud upload)
2. **Offline Access** - No internet needed for basic diagnostics
3. **Cost** - $50 tools vs. $100/visit dealer diagnostics
4. **Learning** - Understand your car, not just fix symptoms
5. **Community** - Contribute knowledge back (forums, GitHub)

### Aligns with r3LAY's Mission
- **Bridge official docs with community knowledge**
- **Garage hobbyist empowerment**
- **Local LLM for private research**
- **Hybrid RAG (indexed + web)**

## Contributing

### How to Add Knowledge
1. **Create markdown file** in `docs/automotive/`
2. **Follow existing format:**
   - Clear headings and structure
   - Practical troubleshooting steps
   - Cost estimates where applicable
   - Community sources cited
3. **Submit PR** or add locally
4. **Re-index** with r3LAY

### Content Guidelines
- ✅ **Practical** - Real-world diagnostic procedures
- ✅ **Cost-conscious** - DIY-first, shop when necessary
- ✅ **Safety-aware** - Note risks (e.g., "Don't drive with failed knock sensor")
- ✅ **Cited** - Link to forums, service manuals, specs
- ❌ **No speculation** - Only verified procedures
- ❌ **No illegal mods** - Emissions compliance matters

## Related Resources

### External (Web)
- **NASIOC** - North American Subaru Impreza Owners Club
- **SubaruForester.org** - Extensive tech articles
- **RomRaider Forums** - Evoscan/SSM1 support
- **YouTube** - ChrisFix, EricTheCarGuy (generic diagnostics)

### Internal (r3LAY)
- **Electronics module** *(coming soon)* - Multimeter use, circuit basics
- **Software module** *(coming soon)* - Evoscan scripting, data analysis

## FAQ

**Q: Will this work for non-Subaru vehicles?**  
A: P0XXX codes are standardized (all OBD2 cars 1996+). P1XXX codes and specific procedures are Subaru-focused, but principles apply broadly.

**Q: Do I need Evoscan for basic diagnostics?**  
A: No. A $20 OBD2 scanner reads codes. Evoscan adds datalogging/performance monitoring (optional).

**Q: Is this safe for beginners?**  
A: Yes. Reading codes and visual inspection are 100% safe. The docs note when professional help is needed.

**Q: How often should I update this knowledge base?**  
A: Add new findings as you encounter them. Community contributions welcome.

**Q: Can I use this commercially (repair shop)?**  
A: This is for personal/hobbyist use. Shops should use professional tools and service manuals.

## License

Content in this module:
- **Generic OBD2 info:** Public domain (SAE standards)
- **Subaru-specific:** Community-contributed knowledge
- **References:** Cited from forums, FSM, community experiences

**Usage:** Free for personal/educational use. No warranty — use at your own risk.

## Acknowledgments

- **NASIOC Community** - Decades of Subaru troubleshooting wisdom
- **RomRaider Developers** - Free Evoscan/SSM1 tools
- **r3LAY Project** - Local-first research platform
- **Garage hobbyists everywhere** - Keeping cars running without dealer markup

---

**Built for:** 1997 Subaru Impreza owners who wrench their own cars  
**Extended to:** Anyone doing OBD2 diagnostics on a budget  
**Philosophy:** Knowledge should be free, local, and private

**Last Updated:** 2026-02-19  
**Version:** 1.0 (Initial prototype)
