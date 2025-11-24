"""
Augment real Kaggle dataset with synthetic data
The real dataset is too small (105 samples), so we'll:
1. Use real data as seed/reference
2. Generate additional realistic samples based on patterns in real data
3. Create a hybrid dataset that's both realistic and sufficient for ML
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def augment_dataset(real_data_path, target_samples=800):
    """
    Augment real dataset with synthetic data based on real patterns

    Args:
        real_data_path: Path to real dataset
        target_samples: Target number of total samples
    """
    print("="*70)
    print(" "*15 + "AUGMENTING REAL DATASET")
    print("="*70 + "\n")

    # Load real data
    df_real = pd.read_csv(real_data_path)
    print(f"Loaded {len(df_real)} real samples")
    print(f"Real careers: {df_real['Career'].nunique()} unique careers")

    # Problem: Too many unique careers (104) with only 1-2 samples each
    # Solution: Group similar careers into broader categories

    # Define career groups (consolidate 104 careers into ~15 main categories)
    career_mapping = {
        # Tech & IT
        'Software Developer': 'Software Engineer',
        'Web Developer': 'Software Engineer',
        'Game Developer': 'Software Engineer',
        'IT Support Specialist': 'IT Specialist',
        'IT Project Manager': 'IT Specialist',
        'Database Administrator': 'Data Analyst',
        'Database Analyst': 'Data Analyst',
        'Data Analyst': 'Data Analyst',
        'Software Quality Assurance Tester': 'Software Engineer',

        # Engineering
        'Mechanical Engineer': 'Engineer',
        'Biomedical Engineer': 'Engineer',
        'Industrial Engineer': 'Engineer',
        'Environmental Engineer': 'Engineer',
        'Aerospace Engineer': 'Engineer',
        'Electrical Engineer': 'Engineer',
        'Civil Engineer': 'Engineer',
        'Robotics Engineer': 'Engineer',
        'Electronics Design Engineer': 'Engineer',
        'Mechanical Designer': 'Engineer',
        'Construction Engineer': 'Engineer',

        # Healthcare
        'Nurse': 'Healthcare Professional',
        'Physician': 'Healthcare Professional',
        'Pediatrician': 'Healthcare Professional',
        'Pediatric Nurse': 'Healthcare Professional',
        'Physical Therapist': 'Healthcare Professional',
        'Dental Hygienist': 'Healthcare Professional',
        'Occupational Therapist': 'Healthcare Professional',
        'Speech Therapist': 'Healthcare Professional',
        'Speech Pathologist': 'Healthcare Professional',
        'Pharmacist': 'Healthcare Professional',
        'Chiropractor': 'Healthcare Professional',
        'Radiologic Technologist': 'Healthcare Professional',
        'Rehabilitation Counselor': 'Healthcare Professional',

        # Education
        'Teacher': 'Teacher',
        'Elementary School Teacher': 'Teacher',

        # Science & Research
        'Research Scientist': 'Research Scientist',
        'Biologist': 'Research Scientist',
        'Zoologist': 'Research Scientist',
        'Forensic Scientist': 'Research Scientist',
        'Geologist': 'Research Scientist',
        'Wildlife Biologist': 'Research Scientist',
        'Astronomer': 'Research Scientist',
        'Biotechnologist': 'Research Scientist',
        'Biomedical Researcher': 'Research Scientist',
        'Marine Biologist': 'Research Scientist',
        'Environmental Scientist': 'Research Scientist',
        'Wildlife Conservationist': 'Research Scientist',
        'Genetic Counselor': 'Research Scientist',

        # Business & Finance
        'Accountant': 'Accountant',
        'Financial Analyst': 'Financial Analyst',
        'Financial Planner': 'Financial Analyst',
        'Financial Advisor': 'Financial Analyst',
        'Financial Auditor': 'Financial Analyst',
        'Investment Banker': 'Financial Analyst',
        'Tax Accountant': 'Accountant',
        'Insurance Underwriter': 'Financial Analyst',

        # Marketing & Sales
        'Marketing Manager': 'Marketing Manager',
        'Marketing Coordinator': 'Marketing Manager',
        'Marketing Analyst': 'Marketing Manager',
        'Marketing Copywriter': 'Marketing Manager',
        'Salesperson': 'Sales Representative',
        'Real Estate Agent': 'Sales Representative',
        'Advertising Executive': 'Marketing Manager',
        'Social Media Manager': 'Marketing Manager',
        'Market Research Analyst': 'Marketing Manager',
        'Market Researcher': 'Marketing Manager',

        # Creative & Design
        'Graphic Designer': 'Graphic Designer',
        'Fashion Designer': 'Graphic Designer',
        'Interior Designer': 'Graphic Designer',
        'Architect': 'Architect',
        'Artist': 'Graphic Designer',
        'Event Photographer': 'Graphic Designer',
        'Fashion Stylist': 'Graphic Designer',

        # Psychology & Counseling
        'Psychologist': 'Psychologist',
        'Forensic Psychologist': 'Psychologist',
        'Marriage Counselor': 'Psychologist',
        'Social Worker': 'Psychologist',

        # HR & Management
        'Human Resources Manager': 'HR Manager',
        'HR Recruiter': 'HR Manager',

        # Legal
        'Lawyer': 'Lawyer',
        'Human Rights Lawyer': 'Lawyer',

        # Media & Communications
        'Journalist': 'Journalist',
        'Technical Writer': 'Journalist',
        'Public Relations Specialist': 'Journalist',

        # Service & Hospitality
        'Chef': 'Chef',
        'Event Planner': 'Event Planner',

        # Other
        'Police Officer': 'Public Service',
        'Police Detective': 'Public Service',
        'Air Traffic Controller': 'Public Service',
        'Airline Pilot': 'Aviation',
        'Forestry Technician': 'Public Service',
        'Video Game Tester': 'Software Engineer',
        'Product Manager': 'Business Analyst',
        'Public Health Analyst': 'Healthcare Professional',
        'Sports Coach': 'Sports Coach',
        'Quality Control Inspector': 'Engineer',
        'Film Director': 'Creative Professional',
        'Diplomat': 'Public Service',
        'Administrative Officer': 'Public Service',
        'Tax Collector': 'Accountant',
        'Foreign Service Officer': 'Public Service',
        'Customs and Border Protection Officer': 'Public Service',
        'Urban Planner': 'Engineer',
    }

    # Apply mapping to consolidate careers
    df_real['Career_Original'] = df_real['Career']
    df_real['Career'] = df_real['Career'].map(career_mapping).fillna(df_real['Career'])

    print(f"\nConsolidated into {df_real['Career'].nunique()} main career categories")
    print("\nNew career distribution:")
    print(df_real['Career'].value_counts())

    # Calculate statistics per career for data generation
    career_stats = {}
    for career in df_real['Career'].unique():
        career_data = df_real[df_real['Career'] == career]
        stats = {}

        for col in df_real.columns:
            if col not in ['Career', 'Career_Original']:
                stats[col] = {
                    'mean': career_data[col].mean(),
                    'std': career_data[col].std() if len(career_data) > 1 else 1.0
                }

        career_stats[career] = stats

    # Generate synthetic samples based on real patterns
    synthetic_samples = []
    samples_needed = target_samples - len(df_real)

    print(f"\nGenerating {samples_needed} synthetic samples based on real patterns...")

    careers_list = list(career_stats.keys())
    samples_per_career = samples_needed // len(careers_list)

    for career in careers_list:
        stats = career_stats[career]

        for _ in range(samples_per_career):
            sample = {'Career': career}

            for feature, feature_stats in stats.items():
                # Generate value with some noise
                value = np.random.normal(feature_stats['mean'], feature_stats['std'] * 1.2)

                # Apply realistic bounds
                if feature in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
                    value = np.clip(value, 1.0, 10.0)
                else:  # Aptitude scores
                    value = np.clip(value, 0.0, 10.0)

                sample[feature] = round(value, 2)

            synthetic_samples.append(sample)

    # Create synthetic DataFrame
    df_synthetic = pd.DataFrame(synthetic_samples)

    # Combine real and synthetic
    df_real_clean = df_real.drop(columns=['Career_Original'])
    df_combined = pd.concat([df_real_clean, df_synthetic], ignore_index=True)

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[SUCCESS] Created hybrid dataset:")
    print(f"  - Real samples: {len(df_real)}")
    print(f"  - Synthetic samples: {len(df_synthetic)}")
    print(f"  - Total: {len(df_combined)}")
    print(f"  - Career categories: {df_combined['Career'].nunique()}")

    print("\nFinal career distribution:")
    print(df_combined['Career'].value_counts())

    # Save
    output_path = Path(real_data_path).parent / 'career_data.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved hybrid dataset to: {output_path}")

    return df_combined


if __name__ == "__main__":
    real_data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'career_data_real.csv'
    augment_dataset(real_data_path, target_samples=800)
