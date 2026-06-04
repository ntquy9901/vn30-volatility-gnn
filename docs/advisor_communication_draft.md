# Draft: Advisor Communication Email

**To:** [Advisor Name]  
**From:** [Your Name]  
**Date:** 2026-06-02  
**Subject:** Critical Bug Discovery in GNNHAR Loss Function — Impact Mitigation Plan

---

Dear [Advisor],

I'm writing to inform you of a critical bug I discovered in the GNNHAR model loss function that affects previous experimental results. I want to be fully transparent about this issue and its impact on my thesis timeline.

## The Issue

**Bug:** Inverted ratio in loss computation  
**Discovery:** 2026-06-01 during systematic code review  
**Root Cause:** Implementation mistake inverted the ratio from `y_true/y_pred` to `y_pred/y_true`

The loss function was computing:
```python
ratio = y_pred / (y_true + eps)  # WRONG - inverted
```

Instead of:
```python
ratio = y_true / (y_pred + eps)  # CORRECT - proper ratio
```

## Impact Assessment

**What's Affected:**
- ❌ All GNNHAR model results (v1.0, v1.1, v1.2)
- ❌ Optuna hyperparameter optimization studies
- ❌ All evaluation metrics from GNNHAR models

**What's NOT Affected:**
- ✅ HAR-RV baseline models (don't use this loss function)
- ✅ Other baseline models (LSTM, MLP, etc.)
- ✅ Thesis theoretical framework and methodology
- ✅ All other aspects of the research

**Timeline Impact:**
- Additional time needed: **+10 days**
- GPU time: ~5 days Optuna + ~3 days model training
- Thesis submission: Still within original timeline buffer

## What I've Done

**Immediate Actions (Completed 2026-06-01):**
1. ✅ Fixed the loss function implementation
2. ✅ Added comprehensive test suite (7 tests, all passing)
3. ✅ Updated documentation and renamed function for clarity
4. ✅ Version tagged as v1.3_LOSS_FIX
5. ✅ Archived invalid results in separate folder
6. ✅ Created full technical documentation

**Quality Assurance:**
- The bug was caught during my systematic review, not during peer review
- This demonstrates research rigor and thorough validation
- All future experiments use corrected, validated implementation

## Recovery Plan

**Week 1 (June 1-7):**
- Re-run Optuna hyperparameter optimization with corrected loss
- Begin training GNNHAR models with corrected hyperparameters

**Week 2 (June 8-14):**
- Complete all GNNHAR model training
- Update thesis results section with v1.3_LOSS_FIX findings only
- Archive comparison data (old vs. new) for appendix if needed

**Week 3 (June 15-21):**
- Finalize all experimental results
- Complete thesis methodology section
- Begin final thesis draft preparation

## Questions for Your Guidance

I would appreciate your advice on the following:

1. **Thesis Documentation:** Should I include a brief "Bug Discovery and Correction" subsection in my methodology section? This could demonstrate research rigor and quality assurance practices.

2. **Experiment Scope:** Given the timeline impact, should I:
   - **Option A:** Focus on key horizons (h=1, h=20) to save time
   - **Option B:** Run full study (all horizons h=1,5,10,20)
   - **Option C:** Your recommendation based on thesis requirements

3. **Results Presentation:** Should I:
   - **Option A:** Include old vs. new comparison in appendix (shows transparency)
   - **Option B:** Focus solely on valid v1.3 results (cleaner presentation)
   - **Option C:** Brief mention in methodology without detailed comparison

## Documentation

Full technical details available at:
- **Bug report:** `docs/bug_fix_v1.3_loss_ratio.md`
- **Test suite:** `gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py`
- **Learning documentation:** `docs/learning/06_gnnhar_ratio_loss.md`

## Silver Lining

While this setback is frustrating, I believe it demonstrates several positive aspects:

1. **Research Rigor:** Systematic code review caught this before submission
2. **Quality Assurance:** Comprehensive test suite prevents future issues
3. **Transparency:** Clear documentation and version control
4. **Learning:** Improved process for all future implementations

The corrected implementation now matches the paper exactly and has been thoroughly validated.

## Timeline Commitment

Even with this setback, I expect to complete the thesis within the original timeline. The additional 10 days are within the buffer I had built into the schedule, and the quality of the final research will be higher for having caught and corrected this issue.

## Next Steps

I'm proceeding with the recovery plan and will keep you updated on progress. I expect to have preliminary v1.3_LOSS_FIX results by our next meeting.

Thank you for your understanding and guidance. Please let me know your thoughts on the questions above.

Best regards,

[Your Name]

---

## Follow-up Notes

**Meeting Date:** [Fill in after advisor meeting]  
**Advisor Feedback:** [Record key points from discussion]  
**Decisions Made:** [Document what was agreed upon]  
**Action Items:** [List any follow-up tasks]

---

*Email draft created: 2026-06-02*  
*Status: Ready to send after review*
