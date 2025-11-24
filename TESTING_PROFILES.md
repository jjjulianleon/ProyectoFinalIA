# Testing Profiles for CareerPath AI

## ⚠️ IMPORTANT: Scale Information

**All aptitudes use 0-10 scale** (NOT 0-100)

- Personality Traits: 1-10
- Aptitude Scores: 0-10

---

## 🧪 Recommended Testing Profiles

### Profile 1: Architect (Spatial Focus)

**Expected Result:** Architect (90%+)

```
Personality (1-10):
- Openness: 8.5
- Conscientiousness: 7.0
- Extraversion: 5.5
- Agreeableness: 5.0
- Neuroticism: 4.5

Aptitudes (0-10):
- Numerical: 6.4
- Spatial: 9.1
- Perceptual: 8.1
- Abstract: 8.7
- Verbal: 4.3
```

---

### Profile 2: Software Engineer (Tech Focus)

**Expected Result:** Software Engineer / Engineer / Research Scientist

```
Personality (1-10):
- Openness: 8.5
- Conscientiousness: 7.0
- Extraversion: 6.0
- Agreeableness: 7.5
- Neuroticism: 4.0

Aptitudes (0-10):
- Numerical: 8.5
- Spatial: 7.5
- Perceptual: 8.0
- Abstract: 8.2
- Verbal: 7.0
```

---

### Profile 3: Healthcare Professional

**Expected Result:** Healthcare Professional / Teacher / Psychologist

```
Personality (1-10):
- Openness: 7.0
- Conscientiousness: 9.0
- Extraversion: 6.0
- Agreeableness: 9.5
- Neuroticism: 3.0

Aptitudes (0-10):
- Numerical: 6.0
- Spatial: 5.0
- Perceptual: 8.0
- Abstract: 6.0
- Verbal: 8.0
```

---

### Profile 4: Graphic Designer (Creative)

**Expected Result:** Graphic Designer / Architect / Creative Professional

```
Personality (1-10):
- Openness: 9.0
- Conscientiousness: 6.0
- Extraversion: 7.0
- Agreeableness: 8.0
- Neuroticism: 5.0

Aptitudes (0-10):
- Numerical: 5.0
- Spatial: 9.0
- Perceptual: 9.0
- Abstract: 7.0
- Verbal: 7.0
```

---

### Profile 5: Marketing Manager (Business)

**Expected Result:** Marketing Manager / Sales Representative / Business Analyst

```
Personality (1-10):
- Openness: 7.5
- Conscientiousness: 7.5
- Extraversion: 9.0
- Agreeableness: 7.0
- Neuroticism: 4.0

Aptitudes (0-10):
- Numerical: 7.0
- Spatial: 5.0
- Perceptual: 7.0
- Abstract: 7.0
- Verbal: 9.0
```

---

### Profile 6: Teacher (Education)

**Expected Result:** Teacher / Healthcare Professional

```
Personality (1-10):
- Openness: 7.0
- Conscientiousness: 8.0
- Extraversion: 7.0
- Agreeableness: 9.0
- Neuroticism: 4.0

Aptitudes (0-10):
- Numerical: 6.0
- Spatial: 5.0
- Perceptual: 7.0
- Abstract: 6.0
- Verbal: 8.5
```

---

### Profile 7: Research Scientist

**Expected Result:** Research Scientist / Engineer / Software Engineer

```
Personality (1-10):
- Openness: 9.0
- Conscientiousness: 8.5
- Extraversion: 4.5
- Agreeableness: 6.0
- Neuroticism: 4.0

Aptitudes (0-10):
- Numerical: 8.5
- Spatial: 7.0
- Perceptual: 8.5
- Abstract: 9.0
- Verbal: 7.0
```

---

### Profile 8: Zero Aptitudes (Edge Case)

**Expected Result:** Sports Coach / Public Service / Musician

```
Personality (1-10):
- Openness: 5.0
- Conscientiousness: 5.0
- Extraversion: 5.0
- Agreeableness: 5.0
- Neuroticism: 5.0

Aptitudes (0-10):
- Numerical: 0.0
- Spatial: 0.0
- Perceptual: 0.0
- Abstract: 0.0
- Verbal: 0.0
```

This tests that low aptitudes don't predict technical careers like Architect or Engineer.

---

## 📊 Validation Checklist

After testing each profile:

- [ ] Top prediction makes sense given the scores
- [ ] High spatial aptitude → Architect/Designer careers
- [ ] High numerical + abstract → Tech/Engineering careers
- [ ] High verbal + extraversion → Marketing/Sales careers
- [ ] High agreeableness + conscientiousness → Healthcare/Teaching
- [ ] Low aptitudes → Non-technical careers

---

## 🔍 Feature Importance Reference

According to the model:

1. **Openness** (11.8%) - Creativity, new experiences
2. **Perceptual Aptitude** (11.3%) - Pattern recognition
3. **Extraversion** (11.1%) - Sociability
4. **Conscientiousness** (10.4%) - Organization
5. **Verbal Reasoning** (9.7%) - Communication

These features have the strongest influence on predictions.

---

## ⚠️ Common Issues

### Issue: Unexpected predictions

**Check:**
1. Are you using 0-10 scale (not 0-100)?
2. Did you enter values for all 10 features?
3. Are personality traits between 1-10?
4. Are aptitudes between 0-10?

### Issue: Architect appearing with low spatial aptitude

**This was a bug!** Fixed in latest version.
- Before fix: Sliders were 0-100
- After fix: Sliders are 0-10
- Now predictions are correct

---

## 🧮 Quick Scale Check

If you see these ranges in Streamlit:
- ✅ Aptitude Scores (0-10) ← **CORRECT**
- ❌ Aptitude Scores (0-100) ← **OLD/WRONG**

If you still see 0-100, pull latest changes:
```bash
git pull
```

---

**Last Updated:** November 2024 (Scale fix applied)
