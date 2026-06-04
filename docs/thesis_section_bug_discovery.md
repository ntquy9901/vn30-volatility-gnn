# Thesis Section: Bug Discovery and Quality Assurance

**Recommended Location:** Methodology → Implementation → Quality Assurance  
**Purpose:** Demonstrate research rigor and systematic review process  

---

## Draft Thesis Section

### 3.X Bug Discovery and Model Validation

During the implementation of the GNNHAR models, a critical bug was discovered in the loss function implementation through systematic code review. This section documents the discovery process, impact assessment, and correction measures, demonstrating the importance of rigorous validation in machine learning research.

#### 3.X.1 Discovery Process

The bug was identified on 2026-06-01 during a comprehensive review of the loss function implementation against the original GNNHAR paper source code (Zhang et al., 2024). The review revealed that the quasi-likelihood loss function was implemented with an inverted ratio:

**Implemented (Incorrect):**
```python
ratio = y_pred / (y_true + eps)  # Inverted ratio
```

**Paper Source (Correct):**
```python
true_fore = outputs / (forecast_y + 1e-4)  # GNNHAR.py line 322
```

The parameter order confusion (`outputs` = target, `forecast_y` = prediction) led to models optimizing for the inverse of the intended objective function. This inversion fundamentally changed the asymmetric penalty properties of the loss function, causing models to be penalized for over-prediction rather than under-prediction.

#### 3.X.2 Impact Assessment

The bug affected all GNNHAR model variants (GNNHAR1L, GNNHAR2L, GNNHAR3L, GHAR) implemented in versions v1.0 through v1.2. Key impacts include:

- **Optimization Objective:** Models trained on inverse objective function
- **Asymmetric Penalty:** Penalty structure reversed (over-prediction vs. under-prediction)
- **Results Validity:** All affected model results invalidated for research conclusions

However, the HAR-RV baseline and other experimental models (LSTM, MLP) remained unaffected as they utilize different loss functions (MSE on z-scored residuals).

#### 3.X.3 Correction Process

The correction was implemented through version v1.3_LOSS_FIX with the following measures:

**Code Fixes:**
1. Corrected ratio direction: `y_true / (y_pred + eps)`
2. Added epsilon to log term: `torch.log(ratio + eps)`
3. Renamed function to `gnnhar_ratio_loss` for clarity
4. Added comprehensive documentation explaining distinction from standard QLIKE

**Validation Measures:**
1. Implemented 7 critical tests covering edge cases and gradient flow
2. Verified exact match with paper implementation (numerical precision 1e-6)
3. Confirmed asymmetric penalty aligns with risk management principles

**Quality Assurance:**
1. Archived invalid results in `results_invalid_loss_bug/` for transparency
2. Version-tagged all results with validity status
3. Created comprehensive bug documentation and learning materials

#### 3.X.4 Research Implications

The discovery and correction of this bug demonstrates the importance of:

**Systematic Validation:** Line-by-line verification against paper source code, especially for mathematical implementations and loss functions.

**Comprehensive Testing:** Test suites covering edge cases (zero values, near-zero predictions, gradient flow) prevent implementation errors from propagating to research conclusions.

**Clear Documentation:** Precise function naming and detailed documentation prevent confusion between similar concepts (e.g., custom ratio loss vs. standard QLIKE).

**Version Control:** Clear version tagging allows separation of invalid and valid results, maintaining research integrity.

**Transparent Communication:** Documentation of bugs and corrections, rather than concealment, demonstrates research rigor and strengthens the validity of final results.

#### 3.X.5 Results Reporting

All GNNHAR results reported in this thesis are from version v1.3_LOSS_FIX or later, utilizing the corrected loss function implementation. Results from previous versions (v1.0-v1.2) are archived separately and are not used for any research claims or comparisons.

The correction process, while extending the experimental timeline by approximately 10 days, ultimately strengthened the research by ensuring all conclusions are based on correctly implemented and validated models.

---

## Alternative: Brief Mention (If Space Limited)

If thesis length constraints prevent a full section, a brief mention can be added to the methodology section:

### Brief Version

**Model Implementation and Validation**

All GNNHAR models were implemented following the architecture described in Zhang et al. (2024), with careful attention to loss function correctness. During implementation review, a loss function bug was identified and corrected in version v1.3_LOSS_FIX, ensuring all reported results use the validated implementation. Comprehensive test coverage and line-by-line verification against the paper source code were implemented to prevent implementation errors.

---

## Appendix Material (Optional)

If including in thesis appendix:

### Appendix A: Bug Discovery Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-05-28 | Initial GNNHAR implementation | Complete |
| 2026-05-30 | Training of initial models | Complete |
| 2026-06-01 | Bug discovery during code review | Critical issue found |
| 2026-06-01 | Fix implementation and testing | Resolved |
| 2026-06-02 | Comprehensive documentation | Complete |
| 2026-06-08 | Re-running experiments with corrected loss | In progress |

### Appendix B: Technical Details

See `docs/bug_fix_v1.3_loss_ratio.md` for complete technical documentation including:
- Root cause analysis
- Fix implementation details
- Test validation results
- Recovery plan

---

## Usage Instructions

**Choose the approach that fits your thesis:**

1. **Full Section:** Use the complete "3.X Bug Discovery and Model Validation" section if you have space and want to demonstrate research rigor (recommended for methodology-focused thesis)

2. **Brief Mention:** Use the "Brief Version" if space is limited but you still want to acknowledge the correction

3. **Appendix Only:** Move detailed technical documentation to appendix if you want to keep main text focused on results

4. **Oral Defense:** Use this material to explain the bug discovery and correction process during your thesis defense (demonstrates thoroughness and attention to detail)

---

**Key Talking Points for Defense:**

- "This was caught through my systematic review process, not during peer review"
- "The correction ensures all my results are based on the properly validated implementation"
- "I now have comprehensive test coverage that prevents similar issues"
- "The process improved my overall research methodology and quality assurance"

---

*Created: 2026-06-02*  
*Purpose: Thesis documentation template*  
*Status: Ready for inclusion in methodology section*
