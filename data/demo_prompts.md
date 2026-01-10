# Demo Prompts for Pathway Q&A Assistant

> **Purpose:** These demo questions test three critical behaviors of the internal Pathway Q&A assistant:
> 1. Grounded, cited answers for in-scope questions
> 2. Probing for missing information when query is under-specified
> 3. Fail-closed refusal for out-of-scope questions

---

## Question A: In-Scope (Grounded Answer with Citations)

### Demo Question

> "A 58-year-old patient presents with unintentional weight loss and persistent heartburn. What is the required referral pathway according to QS124?"

### Ideal Response Behavior

- **Recognize eligibility:** Identify that the patient is ≥55, has weight loss, and has reflux/dyspepsia symptoms—matching QS124-S2 criteria.
- **Cite the specific statement:** Reference "QS124 Statement 2: Urgent Direct Access Endoscopy for Oesophageal or Stomach Cancer" with the exact eligibility criteria.
- **State the required action:** Urgent direct access upper GI endoscopy.
- **Include timing requirement:** Cite that the test must be performed and results returned within 2 weeks.
- **Note GP responsibility:** Mention that the GP retains responsibility for follow-up and acting on results.

### Expected Citation Format

```
[QS124-S2: Triggers / Eligibility]
[QS124-S2: Required Actions]
[QS124-S2: Timing Requirements]
```

---

## Question B: Under-Specified (Probing for Missing Info)

### Demo Question

> "A patient has abdominal pain and changed bowel habits. Should I order a FIT test?"

### Ideal Response Behavior

- **Identify missing information:** The query does not specify:
  - Patient age (QS124-S3 applies to adults only)
  - Whether there is rectal bleeding (exclusion criterion for FIT pathway)
  - Whether 2WW referral criteria have been assessed
- **Probe with specific questions:** Ask:
  - "Is the patient an adult (18+)?"
  - "Is there any rectal bleeding present?"
  - "Have you assessed whether the patient meets the 2WW suspected cancer pathway referral criteria per NG12 recommendations 1.3.1–1.3.3?"
- **Explain why this matters:** Clarify that FIT is only appropriate for patients who do NOT meet 2WW criteria and do NOT have rectal bleeding.
- **Do NOT provide a definitive answer:** Until the missing information is provided, the assistant should not confirm or deny that FIT is appropriate.
- **Cite relevant eligibility section:** Reference "QS124-S3: Triggers / Eligibility" and "QS124-S3: Exclusions / Not Covered" to explain the decision tree.

### Expected Probe Response Structure

```
Before I can advise on FIT testing, I need to clarify:

1. Is the patient an adult?
2. Is there any rectal bleeding?
3. Have 2WW referral criteria (NG12 1.3.1–1.3.3) been assessed?

[QS124-S3: Triggers / Eligibility] — FIT applies only to adults with 
colorectal symptoms who do NOT meet 2WW criteria.
```

---

## Question C: Out-of-Scope (Fail-Closed Refusal)

### Demo Question

> "The patient's FIT came back positive at 85 µg Hb/g faeces. What treatment should I start while waiting for colonoscopy?"

### Ideal Response Behavior

- **Recognize out-of-scope request:** The question asks about treatment, which is explicitly excluded from QS124 and this SOP.
- **Refuse clearly and politely:** State that treatment decisions are outside the scope of this pathway guidance.
- **Cite the scope limitation:** Reference the "Scope: What This Document Does NOT Cover" section, specifically noting that treatment protocols, medication dosing, and patient-specific clinical decisions are excluded.
- **Do NOT speculate or provide treatment advice:** Even if the assistant has knowledge of relevant treatments, it must refuse.
- **Offer to help with in-scope queries:** Suggest alternative questions the assistant can answer (e.g., escalation pathway, documentation requirements, referral process).

### Expected Refusal Response Structure

```
I cannot provide treatment recommendations. This is outside the scope 
of QS124 and this internal pathway SOP.

[Scope: What This Document Does NOT Cover] explicitly excludes:
- Cancer treatment protocols or regimens
- Medication dosing or prescribing guidance
- Patient-specific clinical decision-making

For the patient's positive FIT result, I can help with:
- Escalation and referral pathway (colonoscopy referral)
- Documentation requirements
- What information to provide to the patient

Would you like guidance on any of these in-scope topics?
```

---

## Summary: Expected Behaviors by Question Type

| Question Type | Key Behavior | Citation Requirement |
|---------------|--------------|---------------------|
| **In-Scope** | Grounded answer using statement text | Cite statement ID + section |
| **Under-Specified** | Probe for missing info before answering | Cite eligibility/exclusion sections |
| **Out-of-Scope** | Fail-closed refusal | Cite Scope limitations |

---

## Additional Test Scenarios (Optional)

### Boundary Test: Edge of Scope

> "What are the recommended FIT tests for colorectal cancer screening?"

**Expected behavior:**
- Answer for the **diagnostic pathway** (OC Sensor, HM-JACKarc, FOB Gold per QS124-S3)
- Clarify that NHS Bowel Cancer **Screening Programme** protocols are out of scope

### Ambiguity Test: Age Threshold

> "A patient aged 54 has dyspepsia and weight loss. Do they need urgent endoscopy?"

**Expected behavior:**
- Recognize age criterion is NOT met (requires ≥55 for weight loss + dyspepsia pathway)
- Probe whether dysphagia is present (which would trigger referral at any age)
- Cite QS124-S2 eligibility criteria explicitly

### Documentation Test

> "What do I need to document when making a 2WW referral for suspected cancer?"

**Expected behavior:**
- Answer based on QS124-S4 documentation requirements
- Cite specific elements: recorded discussion, written information provided
- Note that content requirements are listed in QS124-S4 Definitions

