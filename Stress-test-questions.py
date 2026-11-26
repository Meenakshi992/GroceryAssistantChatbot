import textwrap

REPORT_TITLE = "GROCERY-PRODUCTS-ASSISTANT - VERIFIED PRESENTATION TEST REPORT"
SUBTITLE = "Surrey, BC RAG Chatbot | Top-K = 4 | Similarity Cutoff = 0.35 | ChromaDB: chroma_rag_db_GAC"

TEST_CASES = [
    (
        "Test 1 - Direct Fact Lookup [PASSED]",
        "Query: How much is a 10 kg bag of Rogers Granulated White Sugar at No Frills?",
        "Proves: Single-document semantic retrieval & exact inline citation.",
        "Result: Returns exact price with [Source: no_frills_weekly_flyer.pdf, p. 1].",
    ),
    (
        "Test 2 - Direct Fact Lookup (Central City Update) [PASSED]",
        "Query: What is the price of Tilda Pure Basmati Rice (4.5 kg) at Walmart Central City Mall?",
        "Proves: Updated Central City Mall store location & product size match.",
        "Result: Returns exact price & Central City Mall address [Source: walmart_surrey_central_flyer.pdf, p. 1].",
    ),
    (
        "Test 3 - Paraphrased Policy Lookup [PASSED]",
        "Query: How long do I have to use a raincheck before it expires, and where can I redeem it?",
        "Proves: Semantic matching ('before it expires' -> 'valid for 14 days').",
        "Result: Explains 14-day window at issuing store + PC Optimum points [Source: surrey_price_match_and_raincheck_policy.pdf, p. 1].",
    ),
    (
        "Test 4 - Spelling Typos Handling [PASSED]",
        "Query: Wich store near me has the lowest cost on basmatty rice, spagetti noodles, and canola oil?",
        "Proves: Dense embeddings (text-embedding-3-small) handle typos above 0.35 cutoff.",
        "Result: Retrieves Tilda Basmati Rice, Barilla Spaghetti, and Great Value Canola Oil with citations.",
    ),
    (
        "Test 5 - Multi-Store Comparison & Visual Chart [PASSED]",
        "Query: Compare prices of Dairyland 2% Milk, Large White Eggs, Chicken Breasts, and Bananas across nearby stores.",
        "Proves: Multi-document synthesis, per-item tables, and Matplotlib Gap: $X.XX bar chart.",
        "Result: Renders separate per-item tables + grouped bar chart with savings gap badges and citations.",
    ),
    (
        "Test 6 - 'While Quantities Last' Raincheck Trap [PASSED]",
        "Query: If No Frills runs out of No Name Apples before I get there, can I get a raincheck and earn PC Optimum points?",
        "Proves: Balanced retrieval (3 flyer chunks + 2 policy chunks) catches item-level exclusions.",
        "Result: Correctly answers No - No Name Apples are marked 'while quantities last', which is excluded from rainchecks.",
    ),
    (
        "Test 7 - Non-Participating Store Trap [PASSED]",
        "Query: Can I bring the No Frills flyer to Walmart at Central City Mall to get a price match at the register?",
        "Proves: Cross-checks store participation rules in the Surrey policy PDF.",
        "Result: Correctly answers No - Walmart Central City Mall does not offer competitor price matching at the register.",
    ),
    (
        "Test 8 - Loyalty Price Exclusion Trap [PASSED]",
        "Query: Can I price match Safeway's Scene+ member price or Save-On-Foods' More Rewards price at Superstore?",
        "Proves: Catches fine-print Ad Match exclusions in the policy document.",
        "Result: Correctly answers No - member-only loyalty prices (Scene+ / More Rewards) are excluded.",
    ),
    (
        "Test 9 - Freshness Guarantee Exclusion Trap [PASSED]",
        "Query: I bought a pre-packaged salad kit four days ago and its Best Before date passed yesterday - is it covered?",
        "Proves: Policy exception reasoning on the 7-day Fresh Produce Guarantee.",
        "Result: Correctly answers No - pre-packaged salad kits past their Best Before date are excluded.",
    ),
    (
        "Test 10 - Surrey Postal Code & Radius Filter [PASSED]",
        "Input: Set Postal Code = V3T 2W1 (Surrey Central) & Radius = 1.5 km, then ask for milk & banana prices.",
        "Proves: ChromaDB metadata 'where' filter restricts search strictly by travel distance.",
        "Result: Includes only Walmart (Central City) & Superstore; excludes stores outside 1.5 km.",
    ),
    (
        "Test 11 - Invalid Postal Code Validation [PASSED]",
        "Input: Enter '12345' or 'V6B 1A1' (Vancouver) in the Surrey Postal Code input.",
        "Proves: Input validation blocks malformed or non-Surrey postal codes cleanly.",
        "Result: Displays clear red validation error listing valid Surrey FSA prefixes (V3R, V3T, V3V, etc.).",
    ),
    (
        "Test 12 - Out-of-Scope Hallucination Guardrail [PASSED]",
        "Query: What are the SkyTrain Expo Line operating hours from Surrey Central to Waterfront?",
        "Proves: Cosine similarity cutoff (0.35) blocks unrelated questions before LLM generation.",
        "Result: Similarity falls below 0.35; returns safe fallback message with zero hallucination.",
    ),
]


def build_lines():
    lines = [REPORT_TITLE, SUBTITLE, "=" * 92, ""]
    for header, q_line, proves_line, res_line in TEST_CASES:
        lines.append(header)
        for block in (q_line, proves_line, res_line):
            for wrapped in textwrap.wrap(block, width=90):
                lines.append("  " + wrapped)
        lines.append("-" * 92)
    return lines


def save_multipage_pdf(filename, all_lines, lines_per_page=48):
    pages = [all_lines[i:i + lines_per_page] for i in range(0, len(all_lines), lines_per_page)]
    num_pages = len(pages)

    objects = []
    # Object 1: Catalog, Object 2: Pages
    page_obj_nums = [3 + i * 2 for i in range(num_pages)]
    font_obj_num = 3 + num_pages * 2

    kids_str = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("ascii"))

    for idx, page_lines in enumerate(pages):
        page_num = page_obj_nums[idx]
        content_num = page_num + 1

        stream_cmds = ["BT", "/F1 9.5 Tf", "42 750 Td", "13.5 TL"]
        for line in page_lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({safe}) Tj")
            stream_cmds.append("T*")
        stream_cmds.append("ET")

        stream_bytes = "\n".join(stream_cmds).encode("latin-1", errors="replace")

        page_obj = (
            f"{page_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_num} 0 R /Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>\n"
            f"endobj\n"
        ).encode("ascii")

        content_obj = (
            f"{content_num} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

        objects.append(page_obj)
        objects.append(content_obj)

    font_obj = (
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n"
    ).encode("ascii")
    objects.append(font_obj)

    pdf_data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objects:
        offsets.append(len(pdf_data))
        pdf_data.extend(obj)

    xref_offset = len(pdf_data)
    pdf_data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        pdf_data.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf_data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )

    with open(filename, "wb") as f:
        f.write(pdf_data)


if __name__ == "__main__":
    report_lines = build_lines()
    save_multipage_pdf("Presentation_Test_Report.pdf", report_lines)
    print("Created: Presentation_Test_Report.pdf")
