"""
evaluation/build_test_set.py

Builds evaluation/eval_test_set.csv from the chunks_metadata.csv that
Person A exports (evaluation/chunks_metadata.csv).

Output columns
--------------
contract_name    : filename of the contract (510 unique values)
clause_category  : one of the 41 CUAD clause categories (short name, no prefix)
question         : natural-language question for that category
ground_truth     : the expected answer (clause text or Yes/No/date/entity)
chunk_id         : row index in chunks_metadata — used by retrieval_eval.py
                   to identify the ground-truth chunk

The 41 CUAD questions are hard-coded here so the test set can be built
without access to the original CUAD_v1.json.

Run
---
    python evaluation/build_test_set.py
    # writes evaluation/eval_test_set.csv
"""

import os
import pandas as pd

# ── 41 CUAD questions (category → natural language question) ─────────────────
# Source: CUAD_v1.json qas[*].question fields (same for every contract)

CUAD_QUESTIONS: dict[str, str] = {
    "Document Name":
        "What is the name of the contract?",
    "Parties":
        "Who are the parties to the agreement?",
    "Agreement Date":
        "What is the date of the agreement?",
    "Effective Date":
        "On what date does the contract or amendment become effective?",
    "Expiration Date":
        "On what date does the initial term of the contract expire?",
    "Renewal Term":
        "What is the length of the renewal term?",
    "Notice to Terminate Renewal":
        "How many days notice is required to terminate renewal?",
    "Governing Law":
        "Which state/country's law governs the interpretation of the contract?",
    "Most Favored Nation":
        "Is there a most favored nation clause?",
    "Non-Compete":
        "Is there a non-compete clause?",
    "Exclusivity":
        "Is there an exclusivity clause?",
    "No-Solicit of Customers":
        "Is there a no-solicit of customers clause?",
    "No-Solicit of Employees":
        "Is there a no-solicit of employees clause?",
    "Non-Disparagement":
        "Is there a non-disparagement clause?",
    "Termination for Convenience":
        "Does a party have the right to terminate the contract without cause?",
    "ROFR/ROFO/ROFN":
        "Is there a right of first refusal, right of first offer, or right of first negotiation?",
    "Change of Control":
        "Does the contract contain a change of control clause?",
    "Anti-Assignment":
        "Is consent or notice required to assign the contract?",
    "Revenue/Profit Sharing":
        "Is there a revenue or profit sharing clause?",
    "Price Restriction":
        "Is there a restriction on the ability of a party to set or change prices?",
    "Minimum Commitment":
        "Is there a minimum purchase or financial commitment?",
    "Volume Restriction":
        "Is there a cap on the volume of goods or services?",
    "IP Ownership Assignment":
        "Does intellectual property developed during the contract get assigned to the counterparty?",
    "Joint IP Ownership":
        "Is there a clause addressing joint intellectual property ownership?",
    "License Grant":
        "Does the contract grant a license?",
    "Non-Transferable License":
        "Is the license non-transferable?",
    "Affiliate IP License":
        "Does the contract grant a license to an affiliate?",
    "Unlimited/All-You-Can-Eat License":
        "Is the license unlimited in scope?",
    "Irrevocable or Perpetual License":
        "Is the license irrevocable or perpetual?",
    "Source Code Escrow":
        "Is there a source code escrow clause?",
    "Post-Agreement Restrictions":
        "Are there restrictions after the agreement ends?",
    "Audit Rights":
        "Is there an audit right?",
    "Uncapped Liability":
        "Is a party's liability uncapped upon breach?",
    "Cap On Liability":
        "Is liability capped upon breach?",
    "Liquidated Damages":
        "Is there a liquidated damages clause?",
    "Warranty Duration":
        "What is the warranty duration?",
    "Insurance":
        "Is there an insurance clause?",
    "Covenant Not to Sue":
        "Is there a covenant not to sue?",
    "Third Party Beneficiary":
        "Is there a third-party beneficiary?",
    "Indemnification":
        "Is there an indemnification clause?",
    "Dispute Resolution":
        "What is the dispute resolution mechanism?",
}

# Yes/No categories (the 33 that are not extracted-value types)
YES_NO_CATEGORIES = {
    "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit of Customers", "No-Solicit of Employees", "Non-Disparagement",
    "Termination for Convenience", "ROFR/ROFO/ROFN", "Change of Control",
    "Anti-Assignment", "Revenue/Profit Sharing", "Price Restriction",
    "Minimum Commitment", "Volume Restriction", "IP Ownership Assignment",
    "Joint IP Ownership", "License Grant", "Non-Transferable License",
    "Affiliate IP License", "Unlimited/All-You-Can-Eat License",
    "Irrevocable or Perpetual License", "Source Code Escrow",
    "Post-Agreement Restrictions", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration",
    "Insurance", "Covenant Not to Sue", "Third Party Beneficiary",
    "Indemnification", "Dispute Resolution",
}


def _extract_category(raw_category: str) -> str:
    """Strip the 'ContractName__' prefix produced by Person A's export."""
    if "__" in raw_category:
        return raw_category.split("__", 1)[1]
    return raw_category


def build_test_set(
    metadata_path: str = "evaluation/chunks_metadata.csv",
    output_path:   str = "evaluation/eval_test_set.csv",
) -> pd.DataFrame:
    """
    Read chunks_metadata.csv, expand one row per (contract, clause_category)
    positive example, attach the natural-language question, and write
    eval_test_set.csv.

    Yes/No categories use 'Yes' as ground_truth (the clause *is* present
    since we only keep rows where an answer was found).
    """
    meta = pd.read_csv(metadata_path)

    # Normalise the clause_category column
    meta["clause_category"] = meta["clause_category"].apply(_extract_category)

    # Drop rows where the answer is missing (no positive annotation)
    meta = meta.dropna(subset=["answer"])

    records = []
    for _, row in meta.iterrows():
        cat = row["clause_category"]
        if cat not in CUAD_QUESTIONS:
            # Unknown category — skip (shouldn't happen with CUAD v1)
            continue

        ground_truth = (
            "Yes" if cat in YES_NO_CATEGORIES else str(row["answer"]).strip()
        )

        records.append({
            "contract_name":   row["contract_name"],
            "clause_category": cat,
            "question":        CUAD_QUESTIONS[cat],
            "ground_truth":    ground_truth,
            "chunk_id":        int(row["chunk_id"]),
        })

    test_set = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_set.to_csv(output_path, index=False)

    print(f"Test set written to {output_path}")
    print(f"  Total rows      : {len(test_set):,}")
    print(f"  Unique contracts: {test_set['contract_name'].nunique()}")
    print(f"  Clause categories:")
    for cat, count in test_set["clause_category"].value_counts().items():
        print(f"    {cat:<40} {count:>5}")

    return test_set


if __name__ == "__main__":
    build_test_set()
