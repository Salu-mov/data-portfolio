NYC School SAT Performance Analysis

A concise pandas-based exploration of New York City public school SAT results. This project focuses on math performance, overall SAT rankings, and borough-level score variability.

Project Summary

This analysis answers three core questions:

1️⃣ Top Math-Performing Schools

Defined threshold: ≥ 80% of max math score (≥ 640 / 800)

Output: best_math_schools

Columns: school_name, average_math

Sorted in descending order

2️⃣ Top 10 Schools by Total SAT Score

Created feature: total_SAT = average_math + average_reading + average_writing

Output: top_10_schools

Columns: school_name, total_SAT

Ranked in descending order

3️⃣ Borough with Highest Score Variability

Aggregated statistics per borough:

num_schools

average_SAT

std_SAT

Identified borough with the largest standard deviation in SAT performance

Output: largest_std_dev (single-row DataFrame)

-- Tech Stack --

Python 3

pandas

Jupyter Notebook
