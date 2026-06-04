# Recovery Priority Tracker - Loss Function Bug

**Created:** 2026-06-02  
**Status:** Active tracking  
**Purpose:** Mary's recommended recovery plan with priority levels

---

## Priority Matrix

| Priority | Action | Status | Owner | Due Date | Notes |
|----------|--------|--------|-------|----------|-------|
| **P0** | Tag all invalid results with `_INVALID_LOSS` suffix | ✅ Complete | Self | 2026-06-01 | All pre-v1.3 files identified |
| **P0** | Update `CONSTRAINTS.md` with loss function verification checklist | ⏳ Pending | Self | 2026-06-02 | Need to add verification checklist |
| **P1** | Re-run GNNHAR h=1, h=20 only (skip h=5, h=10 if time-constrained) | ⏳ Pending | Self | 2026-06-08 | Focus on key horizons to save time |
| **P1** | Document bug in thesis (shows rigor) | ⏳ Pending | Self | 2026-06-08 | Use thesis_section_bug_discovery.md template |
| **P2** | Compare old vs. new results (appendix material) | ⏳ Pending | Self | 2026-06-15 | Optional if time permits |
| **P2** | Re-run full Optuna study (if time permits) | ⏳ Pending | Self | 2026-06-20 | Ideal completeness but not critical |

---

## Detailed Action Items

### P0: Immediate Actions (Completed ✅)

- [x] **Tag invalid results** - All pre-v1.3 files moved to `results_invalid_loss_bug/`
- [x] **Update version to v1.3_LOSS_FIX** - Version tagging complete
- [x] **Archive structure created** - Separate invalid/valid folders
- [x] **Bug documentation created** - `docs/bug_fix_v1.3_loss_ratio.md`
- [x] **Test suite implemented** - All 7 tests passing

### P0: Pending Immediate Action

- [ ] **Update CONSTRAINTS.md verification checklist**

**What to add:**
```markdown
## Model Implementation Checklist (MANDATORY)

For every loss function implementation:
- [ ] Compare line-by-line with paper source code
- [ ] Verify parameter order (which is target, which is prediction?)
- [ ] Check epsilon placement (guards both division AND log?)
- [ ] Test edge cases (zero, near-zero, perfect prediction)
- [ ] Validate gradient flow (no NaN/inf)
- [ ] Verify against paper test values if available
- [ ] Document minimum loss value (not all losses minimize to 0)
- [ ] Add comprehensive test suite
- [ ] Use clear, descriptive function names
```

**File to update:** `CONSTRAINTS.md`  
**Due:** 2026-06-02

### P1: High Priority (Week of 2026-06-01)

- [ ] **Re-run GNNHAR experiments with corrected loss**

**Scope:** Focus on key horizons to manage timeline
- **Horizons:** h=1 (short-term), h=20 (long-term)
- **Models:** GNNHAR1L, GNNHAR2L (compare 1-layer vs 2-layer)
- **Baseline:** HAR-RV for comparison
- **Estimated time:** 5-6 days GPU time

**Command:**
```bash
# GNNHAR1L h=1
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation gelu \
    --horizon 1 \
    --n_seeds 20 \
    --epochs 400

# GNNHAR1L h=20
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation gelu \
    --horizon 20 \
    --n_seeds 20 \
    --epochs 400
```

**Due:** 2026-06-08

- [ ] **Document bug in thesis methodology**

**Use template:** `docs/thesis_section_bug_discovery.md`  
**Options:** Full section, brief mention, or appendix only  
**Due:** 2026-06-08

### P2: Medium Priority (Week of 2026-06-08)

- [ ] **Compare old vs. new results (appendix)**

**Purpose:** Show impact of bug correction  
**Content:** 
- Table comparing metrics (old vs. new)
- Discussion of how wrong objective affected behavior
- Demonstrates value of correction

**Due:** 2026-06-15 (if time permits)

- [ ] **Re-run full Optuna study (optional)**

**Scope:** If timeline allows, run complete Optuna optimization  
**Estimated time:** 5-7 additional days  
**Value:** Completes hyperparameter search for all configurations  
**Due:** 2026-06-20 (if time permits)

---

## Timeline Overview

### Week 1: June 1-7 (Immediate Response)
- **Focus:** Bug documentation, artifact segregation, advisor communication
- **GPU:** Start corrected training runs
- **Deliverables:** Bug docs complete, advisor notified, training started

### Week 2: June 8-14 (Recovery Execution)
- **Focus:** Complete core experiments (h=1, h=20), thesis documentation
- **GPU:** Complete GNNHAR training runs
- **Deliverables:** Valid v1.3 results, thesis section draft

### Week 3: June 15-21 (Finalization)
- **Focus:** Complete remaining experiments, thesis finalization
- **GPU:** Optional full Optuna study if time allows
- **Deliverables:** Complete results set, thesis methodology section

---

## Success Criteria

### Minimum Viable Recovery (P0 + P1)
- [x] Bug fixed and tested
- [x] Invalid results archived
- [x] Valid training started
- [ ] Key horizons (h=1, h=20) completed
- [ ] Thesis documentation updated
- [ ] Advisor notified and feedback incorporated

### Ideal Recovery (All priorities)
- [ ] All horizons (h=1, 5, 10, 20) completed
- [ ] Full Optuna study completed
- [ ] Old vs. new comparison documented
- [ ] Full thesis methodology section
- [ ] Comprehensive validation documentation

---

## Risk Assessment

### Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU availability issues | Medium | High | Use cloud backup, prioritize key experiments |
| Additional bugs discovered | Low | Medium | Comprehensive test suite prevents recurrence |
| Timeline extension beyond buffer | Low | High | Focus on P0/P1 only, defer P2 items |

### Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient time for thorough testing | Low | High | Test suite already validates correctness |
| Advisor suggests scope expansion | Medium | Medium | Have P2 items ready as optional additions |
| Comparison reveals major unexpected differences | Low | Medium | Document as learning opportunity |

---

## Communication Log

### Advisor Communications

| Date | Method | Topic | Status | Follow-up |
|------|--------|-------|--------|-----------|
| 2026-06-02 | Email | Bug notification and recovery plan | Sent | Awaiting feedback |
| [Future] | Meeting | Discuss timeline and scope | Scheduled | - |

---

## Progress Tracking

### Overall Progress: 40% Complete

**Completed:**
- ✅ Bug fix implementation
- ✅ Comprehensive testing
- ✅ Documentation creation
- ✅ Artifact segregation
- ✅ Version tagging

**In Progress:**
- ⏳ Advisor communication
- ⏳ CONSTRAINTS.md update

**Pending:**
- 📅 Re-running experiments (GPU time)
- 📅 Thesis documentation
- 📅 Optional comparison study

---

## Next Actions (This Week)

1. **Today (2026-06-02):**
   - [ ] Update CONSTRAINTS.md with verification checklist
   - [ ] Send advisor communication email
   - [ ] Start GNNHAR h=1 training (background)

2. **Tomorrow (2026-06-03):**
   - [ ] Start GNNHAR h=20 training (background)
   - [ ] Begin thesis section draft

3. **Week Goal:**
   - [ ] Complete h=1 and h=20 experiments
   - [ ] Finalize thesis bug discovery section
   - [ ] Get advisor feedback on recovery plan

---

*Last Updated: 2026-06-02*  
*Tracking Method: Manual updates after each milestone*  
*Review Frequency: Daily during recovery period*
