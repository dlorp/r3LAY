# r3LAY Automotive Maintenance Tracking Schema

**Purpose:** Structured maintenance history and scheduling for garage hobbyists  
**Session:** Deep Work Session 1 (23:00 AKST, 2026-02-24)  
**Use Case:** Track dlorp's 1997 Impreza service history, schedule upcoming maintenance  
**Philosophy:** Local-first, privacy-preserving, simple YAML (no database required)

---

## Overview

This schema enables:
1. **Service History Logging** — What was done, when, at what mileage, cost
2. **Upcoming Service Scheduling** — Interval-based reminders (mileage or time)
3. **Parts Tracking** — What parts were used, OEM vs aftermarket
4. **Receipt Storage** — Link to PDF/image receipts
5. **DIY vs Shop Tracking** — Cost comparison over time
6. **Provenance** — Where info came from (manual, forum, TSB, experience)

**File Structure:**
```
.r3lay/
└── knowledge/
    └── automotive/
        └── maintenance/
            ├── log.yaml              # Service history (append-only)
            ├── schedule.yaml         # Upcoming services (updated)
            ├── receipts/             # PDF/image storage
            │   ├── 2024-03-15-oil-change.pdf
            │   └── 2024-06-20-brake-pads.jpg
            └── parts-database.yaml   # Parts used, vendors, costs
```

---

## Schema: Service History Log (`log.yaml`)

### Structure
```yaml
vehicle:
  year: 1997
  make: "Subaru"
  model: "Impreza"
  engine: "EJ22"
  transmission: "5MT"
  vin: "JF1GC2xxxxx"  # Optional, last 7 chars for privacy
  nickname: "Blue Thunder"
  purchase_date: "2020-04-15"
  purchase_mileage: 145000

services:
  - date: "2024-03-15"
    mileage: 182340
    service_type: "Oil Change"
    category: "routine"  # routine | repair | diagnostic | upgrade
    description: "Full synthetic oil change, replaced oil filter"
    parts_used:
      - name: "Subaru OEM Oil Filter"
        part_number: "15208AA15A"
        quantity: 1
        cost: 12.50
        vendor: "RockAuto"
      - name: "Mobil 1 5W-30 Full Synthetic"
        quantity: 4.2  # quarts
        cost: 28.00
        vendor: "Walmart"
    labor_cost: 0.00  # DIY
    shop: "DIY"
    total_cost: 40.50
    receipt_path: "receipts/2024-03-15-oil-change.pdf"
    notes: "Changed in driveway, torqued drain plug to 33 ft-lb"
    next_due:
      mileage: 185340  # +3000 miles
      date: "2024-09-15"  # ~6 months
      reason: "oil_interval"
    provenance: "Owner's manual + Subaru service schedule"

  - date: "2024-06-20"
    mileage: 184120
    service_type: "Brake Pads (Front)"
    category: "repair"
    description: "Replaced front brake pads due to squealing, rotors OK"
    symptoms:
      - "Squealing when braking at low speed"
      - "Pad wear indicator visible through wheel"
    parts_used:
      - name: "Akebono ProACT Ceramic Brake Pads"
        part_number: "ACT1324"
        quantity: 1  # set
        cost: 45.00
        vendor: "RockAuto"
      - name: "Brake Caliper Grease"
        cost: 8.00
        vendor: "AutoZone"
    labor_cost: 0.00  # DIY
    shop: "DIY"
    total_cost: 53.00
    receipt_path: "receipts/2024-06-20-brake-pads.jpg"
    notes: "Rear pads at 50%, no replacement needed. Lubed slide pins."
    diagnostic_codes: []  # None
    related_services:
      - "2023-08-10"  # Prior brake inspection
    next_due:
      mileage: 214120  # +30k miles (estimated)
      reason: "brake_pad_life"
    provenance: "Forum (nasioc.com) + parts research"

  - date: "2023-11-05"
    mileage: 175200
    service_type: "Timing Belt Replacement"
    category: "routine"
    description: "105k service: timing belt, water pump, tensioners, idlers"
    parts_used:
      - name: "Gates Timing Belt Kit with Water Pump"
        part_number: "TCKWP329"
        quantity: 1
        cost: 180.00
        vendor: "RockAuto"
    labor_cost: 400.00
    shop: "Hometown Subaru Specialist"
    total_cost: 580.00
    receipt_path: "receipts/2023-11-05-timing-belt.pdf"
    notes: "Included water pump, cam seals, crank seal. Shop recommended due to complexity."
    diagnostic_codes: []
    next_due:
      mileage: 280200  # +105k miles
      date: "2032-11-05"  # 9 years (105 months)
      reason: "timing_belt_interval"
    provenance: "Subaru service schedule (105k interval)"

  - date: "2024-08-12"
    mileage: 186500
    service_type: "Diagnostic (P0420)"
    category: "diagnostic"
    description: "Check engine light: P0420 catalyst efficiency"
    diagnostic_codes:
      - code: "P0420"
        description: "Catalyst System Efficiency Below Threshold (Bank 1)"
        freeze_frame:
          rpm: 2500
          load: 65
          coolant_temp: 195
          vehicle_speed: 55
          stft: "+2%"
          ltft: "+8%"
    parts_used: []
    labor_cost: 0.00
    shop: "DIY"
    total_cost: 0.00
    notes: "Code likely due to high mileage (186k). Fuel trims OK, no lean condition. Monitoring, not urgent."
    resolution: "Cleared code, monitoring. Will test downstream O2 sensor if returns."
    related_research:
      - "docs/automotive/obd2-codes-p0xxx.md#P0420"
      - "https://forums.nasioc.com/forums/showthread.php?t=123456"
    provenance: "r3LAY automotive knowledge base + freeze frame analysis"

  - date: "2024-10-01"
    mileage: 188200
    service_type: "Coolant Flush"
    category: "routine"
    description: "Flush and refill coolant (5 year interval)"
    parts_used:
      - name: "Subaru Genuine Coolant (concentrate)"
        part_number: "SOA868V9270"
        quantity: 1  # gallon
        cost: 25.00
        vendor: "Subaru Dealer"
      - name: "Distilled Water"
        quantity: 1  # gallon
        cost: 2.00
        vendor: "Walmart"
    labor_cost: 0.00
    shop: "DIY"
    total_cost: 27.00
    receipt_path: "receipts/2024-10-01-coolant-flush.pdf"
    notes: "50/50 mix, burped system 3 times, no leaks. Old coolant was orange (5+ years old)."
    next_due:
      date: "2029-10-01"  # 5 years
      reason: "coolant_interval"
    provenance: "Owner's manual (60-month coolant change)"
```

---

## Schema: Upcoming Services (`schedule.yaml`)

### Structure
```yaml
vehicle:
  current_mileage: 189000  # Updated manually or via OBD2 logger
  last_updated: "2024-11-15"

upcoming:
  - service_type: "Oil Change"
    category: "routine"
    due_mileage: 191340
    due_date: "2025-03-15"
    interval:
      mileage: 3000  # miles
      months: 6
    priority: "routine"  # routine | soon | urgent | overdue
    estimated_cost: 45.00
    last_performed: "2024-09-10"
    last_mileage: 188340
    notes: "Full synthetic, DIY"
    parts_needed:
      - "Oil filter (15208AA15A)"
      - "Mobil 1 5W-30 (4.2 qt)"

  - service_type: "Air Filter Replacement"
    category: "routine"
    due_mileage: 195000
    interval:
      mileage: 15000
    priority: "routine"
    estimated_cost: 20.00
    last_performed: "2024-02-10"
    last_mileage: 180000
    notes: "Check every oil change, replace if dirty"
    parts_needed:
      - "K&N or OEM air filter"

  - service_type: "Brake Fluid Flush"
    category: "routine"
    due_date: "2026-06-20"
    interval:
      months: 24  # 2 years
    priority: "soon"
    estimated_cost: 30.00  # DIY
    last_performed: "2024-06-20"
    notes: "DOT 3 or DOT 4, bleed all 4 corners"
    parts_needed:
      - "DOT 3 brake fluid (32 oz)"

  - service_type: "Transmission Fluid Change"
    category: "routine"
    due_mileage: 215200
    interval:
      mileage: 30000
    priority: "routine"
    estimated_cost: 100.00
    last_performed: "2023-05-15"
    last_mileage: 185200
    notes: "Manual transmission, Subaru Extra-S gear oil"
    parts_needed:
      - "Subaru Extra-S 75W-90 (3.7 qt)"

  - service_type: "Spark Plugs Replacement"
    category: "routine"
    due_mileage: 235200
    interval:
      mileage: 60000  # miles
    priority: "routine"
    estimated_cost: 60.00
    last_performed: "2022-08-01"
    last_mileage: 175200
    notes: "NGK iridium, gap to 0.040in"
    parts_needed:
      - "NGK Iridium IX (4 plugs, BKR6EIX-11)"

  - service_type: "Timing Belt Replacement"
    category: "critical"
    due_mileage: 280200
    due_date: "2032-11-05"
    interval:
      mileage: 105000
      months: 105  # 8.75 years
    priority: "routine"  # Changes to "urgent" when <5k miles or 6 months
    estimated_cost: 600.00
    last_performed: "2023-11-05"
    last_mileage: 175200
    notes: "CRITICAL for EJ22 longevity, include water pump + seals"
    parts_needed:
      - "Gates timing belt kit with water pump"
    warnings:
      - "EJ22 is non-interference (safe if breaks) but avoid highway breakdown"

  - service_type: "Investigate P0420 Code"
    category: "diagnostic"
    priority: "monitor"  # monitor | soon | urgent
    triggered_by: "2024-08-12 diagnostic"
    notes: "Code cleared, monitoring. If returns 3+ times, test downstream O2 sensor."
    estimated_cost: 50.00  # O2 sensor if needed
    parts_needed:
      - "Denso downstream O2 sensor (if code persists)"
```

---

## Schema: Parts Database (`parts-database.yaml`)

Track commonly used parts, vendors, costs over time.

```yaml
# Parts used across multiple services
parts:
  - name: "Subaru OEM Oil Filter"
    part_number: "15208AA15A"
    category: "filter"
    compatible_vehicles:
      - "1990-2001 Subaru Impreza (EJ22)"
      - "1995-1999 Subaru Legacy (EJ22)"
    oem: true
    vendors:
      - name: "RockAuto"
        cost: 12.50
        url: "https://www.rockauto.com/..."
        last_checked: "2024-03-15"
      - name: "Subaru Dealer"
        cost: 18.00
        last_checked: "2024-03-15"
    aftermarket_alternatives:
      - name: "Fram Extra Guard"
        part_number: "PH3593A"
        cost: 8.00
        notes: "Cheaper but lower quality, not recommended"
    usage_history:
      - date: "2024-03-15"
        mileage: 182340
        vendor: "RockAuto"
      - date: "2023-09-10"
        mileage: 179340
        vendor: "RockAuto"

  - name: "NGK Iridium IX Spark Plug"
    part_number: "BKR6EIX-11"
    category: "ignition"
    compatible_vehicles:
      - "1990-2001 Subaru Impreza (EJ22)"
    oem: false  # NGK is OEM equivalent
    quantity_needed: 4  # per service
    gap: "0.040 in"
    vendors:
      - name: "RockAuto"
        cost: 12.00  # per plug
        url: "https://www.rockauto.com/..."
      - name: "AutoZone"
        cost: 15.00
    usage_history:
      - date: "2022-08-01"
        mileage: 175200
        vendor: "RockAuto"
        notes: "Gapped to 0.040in, torqued to 18 ft-lb"

  - name: "Gates Timing Belt Kit with Water Pump"
    part_number: "TCKWP329"
    category: "timing"
    compatible_vehicles:
      - "1990-2001 Subaru Impreza (EJ22)"
    oem: false  # Gates is OEM supplier
    vendors:
      - name: "RockAuto"
        cost: 180.00
        last_checked: "2023-11-05"
    includes:
      - "Timing belt"
      - "Water pump"
      - "Tensioner"
      - "Idler pulley"
      - "Cam seals (2)"
      - "Crank seal"
    usage_history:
      - date: "2023-11-05"
        mileage: 175200
        vendor: "RockAuto"
        installed_by: "Hometown Subaru Specialist"
```

---

## Integration with r3LAY

### 1. Maintenance Panel Widget
Display upcoming services, alert when due.

```python
# r3lay/ui/widgets/maintenance_panel.py
from datetime import datetime, timedelta

class MaintenancePanel(Static):
    def render_upcoming_services(self, schedule: dict, current_mileage: int):
        """Render upcoming services with priority highlighting"""
        services = schedule["upcoming"]
        
        for service in services:
            # Calculate urgency
            if service.get("due_mileage"):
                remaining = service["due_mileage"] - current_mileage
                if remaining < 0:
                    priority = "⚠️ OVERDUE"
                    color = "red"
                elif remaining < 500:
                    priority = "🔴 URGENT"
                    color = "orange"
                elif remaining < 2000:
                    priority = "🟡 SOON"
                    color = "yellow"
                else:
                    priority = "🟢 OK"
                    color = "green"
                
                print(f"{priority} {service['service_type']}: {remaining} miles")
            
            # Time-based services
            if service.get("due_date"):
                due = datetime.fromisoformat(service["due_date"])
                remaining_days = (due - datetime.now()).days
                # Similar priority logic
```

### 2. LLM Context Injection
When user asks maintenance questions, inject relevant service history.

```python
def build_maintenance_prompt(query: str, log: dict, schedule: dict) -> str:
    """Enhance LLM prompt with maintenance context"""
    vehicle = log["vehicle"]
    recent_services = log["services"][-5:]  # Last 5 services
    
    context = f"""Vehicle: {vehicle['year']} {vehicle['make']} {vehicle['model']} ({vehicle['engine']})
Current Mileage: {schedule['vehicle']['current_mileage']}

Recent Services:
"""
    for service in recent_services:
        context += f"- {service['date']} @ {service['mileage']} mi: {service['service_type']}\n"
    
    return f"""{context}

User query: {query}

Provide maintenance advice based on this vehicle's history."""
```

### 3. Automated Reminders
Generate reminders when services due.

```python
def check_due_services(schedule: dict, current_mileage: int) -> list[str]:
    """Return list of services due within threshold"""
    due_soon = []
    
    for service in schedule["upcoming"]:
        if service.get("due_mileage"):
            remaining = service["due_mileage"] - current_mileage
            if 0 < remaining < 1000:  # Within 1000 miles
                due_soon.append(
                    f"{service['service_type']} due in {remaining} miles "
                    f"(est. cost: ${service['estimated_cost']})"
                )
    
    return due_soon
```

---

## Real-World Example: dlorp's 1997 Impreza

### Initial Setup
```bash
# Create maintenance tracking
cd ~/repos/r3LAY
mkdir -p .r3lay/knowledge/automotive/maintenance/receipts

# Copy this schema as template
cp docs/automotive/MAINTENANCE_SCHEMA.md .r3lay/knowledge/automotive/maintenance/README.md

# Create log.yaml with current state
cat > .r3lay/knowledge/automotive/maintenance/log.yaml << 'EOF'
vehicle:
  year: 1997
  make: "Subaru"
  model: "Impreza"
  engine: "EJ22"
  transmission: "5MT"
  nickname: "Blue Thunder"
  purchase_date: "YYYY-MM-DD"  # Fill in actual
  purchase_mileage: XXXXXX     # Fill in actual

services:
  # Add historical services if known
  # Start tracking from now if no history
EOF
```

### Adding New Service
```yaml
# Append to log.yaml after each service
  - date: "2024-12-01"
    mileage: 189500
    service_type: "Oil Change"
    category: "routine"
    description: "Winter oil change, full synthetic"
    parts_used:
      - name: "Mobil 1 5W-30"
        quantity: 4.2
        cost: 28.00
        vendor: "Walmart"
      - name: "Subaru OEM Oil Filter"
        part_number: "15208AA15A"
        cost: 12.50
        vendor: "RockAuto"
    labor_cost: 0.00
    shop: "DIY"
    total_cost: 40.50
    notes: "Changed in garage, 20°F outside"
    next_due:
      mileage: 192500
      date: "2025-06-01"
```

---

## Benefits

**For dlorp:**
- Track all Impreza maintenance in one place (no scattered receipts)
- Predict upcoming costs (budget for timing belt, etc.)
- DIY vs shop cost comparison (validate DIY savings)
- Research integration (link forum threads, TSBs to specific services)

**For r3LAY:**
- Real-world dogfooding (use own tool for actual work)
- Maintenance context enhances diagnostic LLM responses
- Axiom validation (does AUTOMOTIVE-009 hold? Track timing belt decisions)
- Privacy-first (all data local, no cloud uploads)

**Cost Savings Example:**
```
2024 Annual Maintenance (log.yaml totals):
- Oil changes (3): $120 DIY vs $180 shop = $60 saved
- Brake pads: $53 DIY vs $250 shop = $197 saved
- Coolant flush: $27 DIY vs $120 shop = $93 saved

Total 2024 DIY Savings: $350

Shop vs DIY Decision (from log):
- Timing belt: $580 shop (too complex, $400 labor justified)
- Brake pads: $53 DIY (simple, saved $197)
```

---

## Next Steps

1. **Phase 1:** Create template files in `.r3lay/knowledge/automotive/maintenance/`
2. **Phase 2:** Populate `log.yaml` with dlorp's Impreza service history (if known)
3. **Phase 3:** Build maintenance panel widget (display upcoming services)
4. **Phase 4:** Integrate with LLM prompts (inject service context)
5. **Phase 5:** Automate reminders (check mileage via OBD2 logger)

---

**Last Updated:** 2026-02-24 (Session 1, 23:55 AKST)  
**Status:** Schema defined, ready for implementation  
**Next:** Create template files, populate with real Impreza data
