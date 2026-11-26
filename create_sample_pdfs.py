import fitz  # PyMuPDF

documents = {
    "no_frills_weekly_flyer.pdf": [
        """NO FRILLS WEEKLY GROCERY FLYER
Store Location: Surrey Central (10153 King George Blvd, Surrey, BC)
Flyer Effective Dates (This Week): Thursday, November 19 to Wednesday, November 25, 2026

CORE GROCERY STAPLES (THIS WEEK'S DEALS)
- Dairyland 2% Partly Skimmed Milk (4L Jug): $5.39 each (Limit 2 per customer).
- No Name Large White Eggs (18 Pack Carton): $4.99 each.
- Boneless Skinless Chicken Breasts (Fresh Club Pack): $11.49 / kg ($5.21 / lb).
- Lean Ground Beef (Fresh Family Pack): $13.99 / kg ($6.35 / lb).
- Fresh Cavendish Bananas (Product of Ecuador): $1.48 / kg ($0.67 / lb).
- Farmer's Market Ambrosia Apples (3 lb Bag, Canada Fancy Grade): $4.49 each.
- Dempster's 100% Whole Wheat Bread (675 g Loaf): $2.97 each.
- No Name Cheddar Cheese Bar (400 g): $4.79 each.

ADDITIONAL WEEKLY SPECIALS
- Eye of Round Roast (Canada AA Grade Beef, while quantities last): $12.96 / kg.
- Rogers Granulated White Sugar (10 kg Bag): $13.99 each."""
    ],
    "walmart_guildford_flyer.pdf": [
        """WALMART SUPERCENTRE WEEKLY GROCERY FLYER
Store Location: Guildford Town Centre (10355 152 St, Surrey, BC)
Flyer Effective Dates (This Week): Thursday, November 19 to Wednesday, November 25, 2026

CORE GROCERY STAPLES (THIS WEEK'S ROLLBACKS)
- Dairyland 2% Partly Skimmed Milk (4L Jug): $5.47 Rollback Price (Limit 2 per customer).
- Great Value Large White Eggs (18 Pack Carton): $5.24 Everyday Low Price.
- Prime Boneless Skinless Chicken Breasts (Fresh Club Pack): $11.98 / kg ($5.43 / lb).
- Lean Ground Beef (Fresh Family Pack): $14.48 / kg ($6.57 / lb).
- Fresh Cavendish Bananas: $1.50 / kg ($0.68 / lb) Weekend Doorcrasher.
- Ambrosia Apples (3 lb Bag, Product of BC): $4.97 per bag.
- Dempster's 100% Whole Wheat Bread (675 g Loaf): $3.00 each (or 2 for $6.00).
- Black Diamond Cheddar Cheese Bar (400 g): $4.97 Member Deal.

ADDITIONAL WEEKLY SPECIALS
- Tilda Pure Basmati Rice (4.5 kg Bag): $12.97 Special Import Price.
- Barilla Classic Pasta (410 g Box): $1.47 each."""
    ],
    "superstore_next_week_flyer.pdf": [
        """REAL CANADIAN SUPERSTORE WEEKLY GROCERY FLYER
Store Location: King George & 104 Ave, Surrey, BC
Flyer Effective Dates (Next Week Preview): Thursday, November 26 to Wednesday, December 2, 2026

CORE GROCERY STAPLES (NEXT WEEK'S FLYER PRICES)
- Dairyland 2% Partly Skimmed Milk (4L Jug): $5.59 each.
- PC Free Run Large White Eggs (18 Pack Carton): $5.79 each.
- PC Air Chilled Boneless Skinless Chicken Breasts (Club Pack): $12.99 / kg ($5.89 / lb).
- Lean Ground Beef (Fresh Club Pack): $14.99 / kg ($6.80 / lb).
- Fresh Cavendish Bananas: $1.65 / kg ($0.75 / lb).
- Farmer's Market Ambrosia Apples (3 lb Bag, Canada Fancy Grade): $4.79 each.
- Dempster's 100% Whole Wheat Bread (675 g Loaf): $3.29 each.
- Cracker Barrel Cheddar Cheese Bar (400 g): $5.49 each.

ADDITIONAL NEXT WEEK SPECIALS
- PC Blue Menu Chicken Thighs (Boneless, Skinless, 696 g): $9.49 each.
- Oikos Greek Yogurt (750 g Tub): $5.49 each."""
    ],
    "price_match_and_raincheck_policy.pdf": [
        """SURREY, BC GROCERY STORES PRICE MATCH & RAINCHECK POLICY
Effective Date: November 2026
Geographic Trade Area: Surrey, BC (Guildford, Surrey Central, Newton, Fleetwood)

1. STORES THAT OFFER AD MATCH / PRICE MATCH IN SURREY
- Participating Stores: No Frills, Real Canadian Superstore, FreshCo, and Save-On-Foods.
- Non-Participating Store: Walmart Supercentre (Guildford Town Centre) does NOT offer price matching, though Walmart's weekly flyer prices can be used to price match at No Frills, Superstore, FreshCo, or Save-On-Foods.

2. AD MATCH RULES & REQUIREMENTS
- The competitor item must be the exact same brand, size, weight, and packaging.
- Customers must show the valid print or digital weekly flyer (e.g., via Flipp or store app) for the active promotion week at checkout.
- Exclusions: Price matching does NOT apply to "While Quantities Last" items, limited-quantity doorcrashers, clearance items, or percentage-off coupons.

3. RAINCHECK & LOYALTY POLICY (NO FRILLS & SUPERSTORE)
- If an advertised flyer item is out of stock during the active Promotion Period, customers may request a Raincheck at Customer Service.
- Rainchecks are valid for 14 days from the issue date at the issuing Surrey store and provide the advertised flyer price plus 10% back in PC Optimum points.
- Rainchecks do NOT apply to items marked "While Quantities Last", gift cards, liquor, or tobacco.
4. While Quantities Last
-It is a retail flyer disclaimer meaning the sale price only applies to the physical stock currently in the store.
"""
    ],
}

for filename, pages in documents.items():
    pdf = fitz.open()
    for page_text in pages:
        page = pdf.new_page()
        rect = fitz.Rect(50, 50, 545, 790)
        page.insert_textbox(rect, page_text, fontsize=11, fontname="helv")
    pdf.save(filename)
    pdf.close()
    print(f"Created: {filename}")
