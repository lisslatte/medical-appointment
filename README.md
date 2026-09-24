# Medical Appointment No-Show Analysis

Interactive analysis of the Kaggle medical appointment no-show dataset, focused on
factors that predict missed appointment rates by category.

www.github.com/lisslatte/medical-appointment

## Project files

- `medical_no_show_analysis.ipynb` - executed Jupyter notebook with data cleaning,
  feature engineering, category analysis, and a logistic no-show risk model.
- `medical_no_show_story_dashboard.html` - standalone interactive story dashboard.
- `index.html` - same dashboard, named for easy GitHub Pages publishing.
- `build_medical_appointment_analysis.py` - reproducible builder for the notebook,
  dashboard, and exported analysis tables.
- `analysis_outputs/` - CSV and JSON outputs used for review and reuse.
- `noshowappointments-kagglev2-may-2016.csv` - source dataset.

## Main findings

- Overall no-show rate is about 20%.
- Appointment lead time is the strongest practical signal: same-day appointments
  have much lower no-show rates than appointments booked weeks ahead.
- Teen and young-adult patients have higher observed no-show rates than older
  adult groups.
- Scholarship patients have higher observed no-show rates, suggesting a need for
  access support rather than punitive policies.
- SMS reminder results should be interpreted carefully because reminder receipt
  is confounded by appointment timing and other operational rules.

## Recommended institute actions

- Shorten lead times where possible and fill cancellations from a waitlist.
- Use two-way reminders with easy confirmation and rescheduling.
- Add support for higher-risk groups, including transport guidance, flexible
  slots, caregiver contact options, and low-friction rescheduling.
- Monitor neighborhood hot spots and adapt outreach by local access barriers.
- Use risk scoring to add support, not to restrict access to care.

## Rebuild

```bash
python build_medical_appointment_analysis.py
```

