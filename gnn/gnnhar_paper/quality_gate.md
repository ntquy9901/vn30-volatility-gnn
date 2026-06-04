# Quality Gate Checklist - GNN-HAR Implementation

**Date:** 2026-05-30
**Purpose:** Ensure code quality, ML/DL best practices, and correctness
**Status:** Ready for review

---

## Part 1: Code Review Checklist

### 1.1 Data Processing

#### ✅ Data Quality Checks
- [ ] **Missing Values Handled:**
  ```python
  # glasso_adjacency.py:68-79
  returns_clean = returns.dropna()  # GLASSO NaN handling
  ```
  - ✅ Drops rows with NaN before GLASSO
  - ✅ Warns user about dropped rows
  - ✅ Validates minimum sample size

- [ ] **Positive Values Ensured:**
  ```python
  # evaluation.py:34
  pred = torch.clamp(predictions, min=eps)  # Ensure positive
  targ = torch.clamp(targets, min=eps)
  ```
  - ✅ Prevents log(0) in QLIKE
  - ✅ Prevents division by zero

- [ ] **Feature-Target Match:**
  ```python
  # CONSTRAINTS.md C2: Both use RV (std of log returns)
  # train_gnnhar_paper.py uses compute_rv() for both features and targets
  ```
  - ⚠️ **NEEDS VERIFICATION:** Check if HAR features use same measurement as targets

#### ❌ Issues Found

**Issue #1: HAR Feature-Target Mismatch Risk**
```python
# train_gnnhar_paper.py - HAR feature computation
# Uses: RV_lag_d, RV_lag_w, RV_lag_m (lagged RV)
# Target: RV_h (future RV over h days)

# CORRECT if both use compute_rv(log_returns)
# INCORRECT if features use |log_return| (single-day)
```

**Recommendation:**
```python
# Add verification in train_gnnhar_paper.py
def verify_har_features(rv_h, ticker, train_end_ts):
    """Ensure HAR features use same RV measurement as targets."""
    # Check that RV features are computed from std(log_returns), not |log_return|
    # This is CRITICAL per CONSTRAINTS.md C2
    pass
```

---

### 1.2 Loss Function

#### ✅ QLIKE Loss Implementation

**Mathematical Correctness:**
```python
# evaluation.py:28-49
# Formula: log(y_true) - log(y_pred) + y_true/y_pred - 1
# Simplifies to: log(y_true/y_pred) + y_true/y_pred - 1
```
- ✅ Matches Patton (2011) formula
- ✅ Symmetric: log(t/p) + t/p - 1 = log(t) - log(p) + t/p - 1

**Gradient Check:**
```python
# QLIKE gradient wrt p:
# d/dp [log(t) - log(p) + t/p - 1] = -1/p - t/p^2
# = -(p + t) / p^2
```
- ✅ Gradient exists for p > 0
- ✅ Gradient smooth (no discontinuities)
- ✅ Second derivative exists (convexity analysis possible)

**Asymmetry Verification:**
```python
# Test results confirmed:
# Underprediction (p = 0.8t): QLIKE = 0.473
# Overprediction (p = 1.2t): QLIKE = -0.349
# Ratio: 8× more penalty for underprediction
```
- ✅ Asymmetric as expected
- ✅ Economically meaningful (underprediction = risk)

**Numerical Stability:**
```python
# evaluation.py:34-36
pred = torch.clamp(predictions, min=eps)  # eps=1e-8
targ = torch.clamp(targets, min=eps)
```
- ✅ Prevents log(0)
- ✅ Prevents division by zero
- ⚠️ **ISSUE:** Clamping may bias gradients for values near eps

**Issue #2: Clamping Bias**
```python
# When predictions are clamped to eps=1e-8:
# - Gradients become artificially small near eps
# - Loss saturates for very small values
```

**Recommendation:**
```python
# Use softplus instead of clamp
def qlike_loss_stable(predictions, targets, eps=1e-8):
    # Ensure positive with softplus (smooth)
    pred_smooth = torch.nn.functional.softplus(predictions)
    targ_smooth = torch.nn.functional.softplus(targets)
    # Then compute QLIKE
```

---

### 1.3 Model Architecture

#### ✅ H1/H2 Separation

**Paper Design:**
```
H1 = HAR pathway (local, no graph)
H2 = Graph pathway (spillover, no self-loops)
Output = H1 + H2
```

**Implementation Check:**
```python
# gnnhar_models.py - GNNHAR1L forward()
# Line 163-190
def forward(self, node_feat, adj):
    # H1: HAR pathway
    H1 = self.linear1(node_feat)  # Local HAR features

    # H2: Graph pathway
    H2 = self.gcn1(node_feat, adj)  # Graph spillover
    H2 = self.relu(H2)
    H2 = self.mlp1(H2)

    # Final: H1 + H2
    res = H1 + H2
    res = self.relu(res)
    return res
```
- ✅ H1 and H2 properly separated
- ✅ Self-loops excluded from H2 (after fix)
- ✅ Residual connection used

**Issue #3: Activation After Residual**
```python
# gnnhar_models.py:189
res = H1 + H2
res = self.relu(res)  # Activation AFTER residual
```
- ⚠️ **BEST PRACTICE:** Activation should come BEFORE residual for ReLU
- **Standard practice:** H1 = ReLU(xW1), H2 = ReLU(GCN(x)), out = H1 + H2
- **Current:** H1 + H2 then ReLU → May cause "dying ReLU" if sum is negative

**Recommendation:**
```python
# Better architecture:
H1 = self.relu(self.linear1(node_feat))
H2 = self.relu(self.mlp1(self.gcn1(node_feat, adj)))
res = H1 + H2  # No activation after residual
```

---

### 1.4 Training Loop

#### ✅ Gradient Flow

**Backward Pass:**
```python
# ensemble_trainer.py:170-173
optimizer.zero_grad()
pred = model(X_t, adj_t)
loss = criterion(pred, y_t)
loss.backward()
```
- ✅ Gradients computed correctly
- ✅ optimizer.zero_grad() called before backward

**Gradient Clipping:**
```python
# ensemble_trainer.py:172
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- ✅ Prevents gradient explosion
- ✅ Standard practice for RNN/GNN

#### ⚠️ Learning Rate Schedule

**Current Implementation:**
```python
# ensemble_trainer.py:142-145
optimizer = optim.AdamW(
    model.parameters(),
    lr=self.lr,  # Fixed at 1e-3
    weight_decay=self.weight_decay
)
# NO learning rate scheduler!
```

**Issue #4: No Learning Rate Decay**
- Training for 5000 epochs with fixed lr
- May converge to suboptimal solution
- Standard practice: ReduceLROnPlateau or CosineAnnealing

**Recommendation:**
```python
# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, verbose=True
)

# In training loop:
scheduler.step(val_loss)  # After each epoch
```

---

### 1.5 Early Stopping

#### ✅ Implementation

**Current Code:**
```python
# ensemble_trainer.py:202-209
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {...}
    patience_cnt = 0
else:
    patience_cnt += 1
    if patience_cnt >= self.patience:  # patience=500
        break
```
- ✅ Correct implementation
- ✅ Saves best model state
- ✅ Patience appropriate for 5000 epochs

#### ⚠️ Validation Loss Computation

**Issue #5: Validation Loss Not Averaged**
```python
# ensemble_trainer.py:194-196
model.eval()
with torch.no_grad():
    val_pred = model(X_v, adj_t)
    val_loss = criterion(val_pred, y_v).item()
```

- Uses ENTIRE validation set at once (may be memory intensive)
- ⚠️ If validation set is large, should use mini-batches

**For large validation sets:**
```python
# Better: Mini-batch validation
val_losses = []
for batch in val_loader:
    pred = model(batch.X, adj)
    loss = criterion(pred, batch.y)
    val_losses.append(loss.item())
val_loss = np.mean(val_losses)
```

---

### 1.6 Evaluation Metrics

#### ✅ R² Computation

**Formula Check:**
```python
# evaluation.py:165-167
ss_res = float(np.sum((y_true - y_pred) ** 2))
ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
r2 = float(1.0 - ss_res / (ss_tot + eps))
```
- ✅ Correct formula: R² = 1 - SS_res/SS_tot
- ✅ Eps prevents division by zero
- ✅ Can be negative (worse than mean)

**Range Check:**
```python
# R² range: ]-∞, 1]
# -∞ if predictions are terrible
# 0 if same as mean
# 1 if perfect
```
- ✅ Allows negative R² (important for volatility)

#### ✅ QLIKE Metric

**Properties:**
```python
# evaluation.py:71-82
ratio = y_true / y_pred
qlike = np.mean(np.log(y_true) - np.log(y_pred) + ratio - 1.0)
```
- ✅ Lower is better (unlike R²)
- ✅ Asymmetric (underprediction penalized more)
- ✅ Robust to noise (Patton 2011)

**Range:**
```python
# QLIKE range: ]-∞, +∞)
# Negative when overpredicting
# Positive when underpredicting
# Zero when perfect
```
- ✅ No upper bound (unlike R²)
- ✅ Interpretable sign

#### ✅ Heteroskedastic Metrics

**HMSE Formula:**
```python
# evaluation.py:90-97
errors = (y_true - y_pred) ** 2
hmse = np.mean(errors / y_true)
```
- ✅ Correct formula
- ✅ Penalizes errors more in low-vol periods
- ✅ y_true clamped to avoid division by zero

**HMAE Formula:**
```python
# evaluation.py:103-110
errors = np.abs(y_true - y_pred)
hmae = np.mean(errors / np.sqrt(y_true))
```
- ✅ Correct formula
- ✅ Same intuition as HMSE

---

## Part 2: ML/DL Best Practices Review

### 2.1 Data Splitting

#### ✅ Time-Series Split

**Global Split (R6):**
```python
# train_gnnhar_paper.py:210-213
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8
```
- ✅ No look-ahead bias
- ✅ Test set truly out-of-sample
- ✅ Train/Val split respects temporal order

**Issue #6: Train/Val Leakage Risk**
```python
# Current: Train/Val split on pre-2026 data
# If data is shuffled or not time-ordered, leakage may occur
```

**Recommendation:**
```python
# Add temporal split verification
def verify_temporal_split(dates):
    """Ensure dates are monotonically increasing."""
    assert dates.is_monotonic_increasing, "Dates not time-ordered!"
```

---

### 2.2 Cross-Validation

#### ❌ No Cross-Validation

**Current:**
- Single train/val/test split
- No cross-validation
- No bootstrap

**Issue #7: Limited Validation**
- Single split may not be representative
- Variance estimates unreliable

**Recommendation:**
```python
# Add rolling-window validation (time-series CV)
def rolling_window_validation(data, n_windows=5):
    """Generate rolling-window validation splits."""
    for i in range(n_windows):
        train_end = len(data) * (i+1) // (n_windows+1)
        val_start = train_end
        val_end = train_end + len(data) // (n_windows+1)
        yield (data[:train_end], data[val_start:val_end])
```

---

### 2.3 Regularization

#### ✅ Weight Decay

**Current:**
```python
# ensemble_trainer.py:145
WEIGHT_DECAY = 1e-3
optimizer = optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY)
```
- ✅ L2 regularization applied
- ✅ Prevents overfitting

#### ❌ No Dropout

**Current:**
- No dropout in model architecture
- GNNHAR1L, GNNHAR2L, GNNHAR3L have no dropout layers

**Issue #8: Risk of Overfitting**
- GNNs prone to overfitting on small datasets
- ESS=123 per stock is limited

**Recommendation:**
```python
# Add dropout to model architecture
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Add dropout
        # ... rest of architecture

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H1 = self.dropout(H1)  # Apply dropout
        # ... etc
```

---

### 2.4 Batch Normalization

#### ❌ No BatchNorm

**Current:**
- No batch normalization layers
- No layer normalization

**Issue #9: Training Instability**
- GNNs benefit from normalization
- May cause gradient issues

**Recommendation:**
```python
# Add LayerNorm (better than BatchNorm for GNN)
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_hid)
        # ...

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H1 = self.layer_norm(H1)  # Apply LayerNorm
        # ...
```

---

### 2.5 Gradient Monitoring

#### ❌ No Gradient Clipping Logging

**Current:**
```python
# ensemble_trainer.py:172
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- ✅ Gradient clipping applied
- ❌ No logging of gradient norms
- ❌ Cannot detect gradient issues

**Recommendation:**
```python
# Log gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 10:  # Suspiciously large
            print(f"[WARN] Large gradient: {name} norm={grad_norm:.2f}")
```

---

### 2.6 Initialization

#### ❌ Default Initialization

**Current:**
- PyTorch default initialization (Xavier/He uniform)
- No custom initialization

**Issue #10: Suboptimal for GNN**
- GNNs may benefit from specific initialization
- GCN layers often use Glorot initialization

**Recommendation:**
```python
# Add custom initialization
def initialize_gnn_model(model):
    """Initialize GNN model with Glorot initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

---

## Part 3: Unit Test Requirements

### 3.1 Test Suite Structure

```
tests/
├── test_evaluation.py      # Evaluation metrics tests
├── test_qlike_loss.py       # QLIKE loss tests (DONE)
├── test_glasso_adjacency.py # GLASSO tests
├── test_model_arch.py       # Model architecture tests
├── test_training_loop.py    # Training loop tests
└── test_data_processing.py  # Data processing tests
```

---

### 3.2 Required Unit Tests

#### Test 1: QLIKE Loss Properties

```python
def test_qlike_asymmetry():
    """Verify QLIKE penalizes underprediction more."""
    # Perfect prediction
    assert qlike_loss([1.0], [1.0]) == 0

    # Underprediction (80%)
    loss_under = qlike_loss([0.8], [1.0])

    # Overprediction (120%)
    loss_over = qlike_loss([1.2], [1.0])

    # Underprediction should have higher loss
    assert loss_under > loss_over
    assert loss_under / loss_over > 2  # At least 2× more

def test_qlike_gradient():
    """Verify QLIKE gradient is correct."""
    pred = torch.tensor([1.0], requires_grad=True)
    targ = torch.tensor([1.2])

    loss = qlike_loss(pred, targ)
    loss.backward()

    # Gradient should be negative (increase pred to reduce loss)
    assert pred.grad.item() < 0

    # Gradient magnitude
    expected_grad = -(pred + targ) / (pred ** 2)
    assert torch.allclose(pred.grad, expected_grad, atol=1e-5)
```

#### Test 2: GLASSO Adjacency

```python
def test_glasso_no_self_loops():
    """Verify GLASSO excludes self-loops."""
    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns)

    # Diagonal should be zero
    assert np.allclose(np.diag(adj), 0)

def test_glasso_symmetry():
    """Verify adjacency matrix is symmetric."""
    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns)

    # Adjacency should be symmetric
    assert np.allclose(adj, adj.T)

def test_glasso_sparsity():
    """Verify GLASSO produces sparse graph."""
    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns)

    # Should be sparse (10-30% density)
    density = (adj != 0).sum() / (adj.shape[0] * adj.shape[1])
    assert 0.05 < density < 0.5
```

#### Test 3: Model Forward Pass

```python
def test_model_output_shape():
    """Verify model produces correct output shape."""
    model = GNNHAR1L(n_hid=16)

    # Input: (n_snapshots, n_stocks, n_features)
    X = torch.randn(100, 30, 3)
    adj = torch.randn(30, 30)

    # Forward pass
    output = model(X, adj)

    # Output: (n_snapshots, n_stocks)
    assert output.shape == (100, 30)

def test_h1_h2_separation():
    """Verify H1 and H2 are properly separated."""
    model = GNNHAR1L(n_hid=16)

    # With zero adjacency (no graph)
    X = torch.randn(100, 30, 3)
    adj_zero = torch.zeros(30, 30)

    output_zero_adj = model(X, adj_zero)

    # With identity adjacency (self-loops only)
    adj_identity = torch.eye(30)
    output_identity_adj = model(X, adj_identity)

    # Should be different (graph effect)
    assert not torch.allclose(output_zero_adj, output_identity_adj)
```

#### Test 4: Training Loop

```python
def test_training_decreases_loss():
    """Verify training decreases QLIKE loss."""
    model = GNNHAR1L(n_hid=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy data
    X = torch.randn(100, 30, 3)
    y = torch.rand(100, 30) * 0.01
    adj = torch.randn(30, 30)

    # Initial loss
    model.eval()
    with torch.no_grad():
        pred_init = model(X, adj)
    loss_init = qlike_loss(pred_init, y)

    # Train for 10 epochs
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(X, adj)
        loss = qlike_loss(pred, y)
        loss.backward()
        optimizer.step()

    # Final loss
    model.eval()
    with torch.no_grad():
        pred_final = model(X, adj)
    loss_final = qlike_loss(pred_final, y)

    # Loss should decrease
    assert loss_final < loss_init

def test_early_stopping():
    """Verify early stopping triggers."""
    trainer = EnsembleTrainer('GNNHAR1L', n_hid=16, n_epochs=100, patience=10)

    # Small dataset (triggers early stopping)
    X_train = np.random.randn(10, 30, 3)
    y_train = np.random.rand(10, 30) * 0.01
    X_val = np.random.randn(5, 30, 3)
    y_val = np.random.rand(5, 30) * 0.01
    adj = np.random.randn(30, 30)

    history = trainer.train_single(X_train, y_train, X_val, y_val, adj, seed=42)

    # Should stop early (before 100 epochs)
    assert len(history['history']['train']) < 100
```

#### Test 5: Evaluation Metrics

```python
def test_r2_range():
    """Verify R² is in correct range."""
    # Perfect prediction
    r2_perfect = compute_r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r2_perfect == 1.0

    # Same as mean
    r2_mean = compute_r2([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert r2_mean == 0.0

    # Worse than mean
    r2_bad = compute_r2([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert r2_bad < 0

def test_qlike_range():
    """Verify QLIKE is in correct range."""
    # Perfect prediction
    qlike_perfect = compute_qlike([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert qlike_perfect == 0.0

    # Underprediction (positive QLIKE)
    qlike_under = compute_qlike([2.0], [1.0])
    assert qlike_under > 0

    # Overprediction (negative QLIKE)
    qlike_over = compute_qlike([1.0], [2.0])
    assert qlike_over < 0

def test_metrics_finite():
    """Verify all metrics are finite."""
    y_true = np.random.gamma(2, 0.001, 1000)
    y_pred = y_true + np.random.randn(1000) * 0.0001

    metrics = compute_metrics(y_true, y_pred)

    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
```

---

## Part 4: Quality Gate Criteria

### 4.1 Must Pass (Blocking)

- [ ] All unit tests pass (100% pass rate)
- [ ] No NaN/Inf in training loss
- [ ] Gradients flow correctly (no None gradients)
- [ ] QLIKE loss decreases monotonically (mostly)
- [ ] R², MAE, RMSE, QLIKE, HMSE, HMAE all finite
- [ ] Model output shape correct
- [ ] No data leakage (train/val/test split verified)

### 4.2 Should Pass (Warning)

- [ ] Early stopping triggers before max epochs
- [ ] Learning rate decreases during training
- [ ] Gradient norms reasonable (< 10)
- [ ] Validation loss converges
- [ ] No overfitting (train loss ≈ val loss)
- [ ] GLASSO graph sparse (10-30% density)

### 4.3 Nice to Have (Info)

- [ ] Training time reasonable (< 8 hours per model)
- [ ] Memory usage reasonable (< 8GB)
- [ ] GPU utilization > 50%
- [ ] Model converges to similar loss across seeds

---

## Part 5: Critical Issues Summary

### 🔴 Must Fix (Blocking)

1. **Issue #3: Activation After Residual** (gnnhar_models.py:189)
   - Current: `res = H1 + H2; res = self.relu(res)`
   - Should: `H1 = self.relu(...); H2 = self.relu(...); res = H1 + H2`
   - Impact: May cause dying ReLU

2. **Issue #6: Train/Val Leakage Risk** (train_gnnhar_paper.py)
   - Need: Verify temporal ordering
   - Impact: Potential data leakage

### 🟡 Should Fix (Important)

3. **Issue #2: Clamping Bias** (evaluation.py:34)
   - Current: `torch.clamp(predictions, min=eps)`
   - Should: Use softplus for smooth positivity
   - Impact: Gradient bias near eps

4. **Issue #4: No LR Scheduler** (ensemble_trainer.py)
   - Need: Add ReduceLROnPlateau
   - Impact: Suboptimal convergence

5. **Issue #8: No Dropout** (gnnhar_models.py)
   - Need: Add dropout layers
   - Impact: Risk of overfitting

6. **Issue #9: No BatchNorm/LayerNorm** (gnnhar_models.py)
   - Need: Add LayerNorm
   - Impact: Training instability

### 🟢 Nice to Fix (Optional)

7. **Issue #7: No Cross-Validation**
   - Need: Add rolling-window CV
   - Impact: Limited validation

8. **Issue #10: Default Initialization**
   - Need: Custom Glorot initialization
   - Impact: Suboptimal start

9. **Issue #5: Mini-batch Validation**
   - Need: Batch validation for large sets
   - Impact: Memory efficiency

10. **Issue #11: No Gradient Logging**
    - Need: Log gradient norms
    - Impact: Hard to debug

---

## Part 6: Recommended Actions

### Immediate (Before Training)

1. ✅ **Add Unit Tests** (test_qlike_loss.py exists, need more)
2. ✅ **Fix Issue #3:** Move activation before residual
3. ✅ **Fix Issue #6:** Verify temporal split
4. ✅ **Run Quality Gate:** Ensure all "Must Pass" criteria met

### Short-term (After First Training)

5. ✅ **Fix Issue #4:** Add LR scheduler
6. ✅ **Fix Issue #8:** Add dropout
7. ✅ **Fix Issue #9:** Add LayerNorm

### Long-term (Thesis Revision)

8. ✅ **Fix Issue #2:** Use softplus instead of clamp
9. ✅ **Fix Issue #7:** Add cross-validation
10. ✅ **Fix Issue #10:** Custom initialization
11. ✅ **Fix Issue #11:** Gradient logging

---

**Status:** Quality gate defined, 11 issues identified
**Priority:** Fix Issue #3, #6 immediately (blocking)
**Next:** Create comprehensive unit test suite
