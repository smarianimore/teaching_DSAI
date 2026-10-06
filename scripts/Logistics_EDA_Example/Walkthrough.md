# Logistics EDA: a management case-study walkthrough

Turn a management question into an analytical design :) 
The Python code is the implementation of that design.

**Use this reasoning chain throughout:**

> Management question 
> → decision-relevant KPI 
> → plot that makes the comparison visible 
> → derived metrics and features 
> → original observations and business rules 

Design backward from the question to the required data. 
Implement forward from validated source data to the result. 

A sophisticated method cannot compensate for a poorly defined question, 
an incorrect denominator, or missing evidence.

## The case and the decision

A distribution operation sends orders through carriers A and B on two lanes: Local and Regional. 
Management observes a low delivery service rate and asks whether it should 
change carrier allocation, 
revise service commitments, 
or investigate operational problems.

The dataset covers 90 release days and 6,024 unique orders. 
The observation snapshot is **2 April 2026 at 00:00 UTC**. 
Carrier A receives more Regional orders. 
Regional orders take longer and have a different service commitment. 
Some records are duplicated, incomplete, or invalid; some orders remain in transit.

These details matter analytically:

- Carrier assignment is associated with work difficulty, 
so raw carrier rankings may be misleading.
- Recent orders have had less time to finish, 
so completed-only analysis can favor fast orders.
- Missing timestamps and true long delays require different treatment.

## How to use the code

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```

Alternatively, open `logistics_eda.ipynb` in JupyterLab and run its cells from the project folder. 
The notebook and script implement the same analysis; either is sufficient. 
Running the pipeline regenerates the simulated inputs and outputs.

| File | Role in the case |
| --- | --- |
| `pipeline.py` | Simulation, data audit, cleaning, feature creation, analysis, plots, report |
| `logistics_eda.ipynb` | Notebook version with the same numbered sections |
| `app.py` | Interactive app example |
| `outputs/orders_raw.csv` | Order extract with intentionally injected problems |
| `outputs/reference.sqlite` | Lane commitments in a reference table |
| `outputs/orders_clean.csv` | Auditable order-level analysis dataset |
| `outputs/quality_audit.csv` | Counts documenting data quality and cohort eligibility |
| `outputs/report.html` | Generated case report; keep it beside `overview.png` |
| `outputs/versions.json` | Package versions used for the supplied outputs |

**Implementation labels used below:** 
“Implemented” refers to the supplied pipeline or dashboard. 
Code snippets use variables created by the pipeline and should be run after the stated section. 
They are focused excerpts, not independent scripts.

## Build an analytical "contract"

Before choosing a statistic, specify the population, observation unit, time basis, and outcome rule.

| Contract element | Choice in this project | Why it matters |
| --- | --- | --- |
| Observation grain | One row per order after removing exact duplicates | Counting raw rows would overweight duplicate orders |
| Identifier | `order_id` must be unique | Conflicting versions require reconciliation, not arbitrary deletion |
| Time convention | UTC and elapsed calendar hours | Working days, or daylight-saving rules would require specific handling |
| Deadline | Release timestamp plus lane-specific `sla_h` | A service outcome depends on the promise, not duration alone |
| Service cohort | Orders whose deadlines have passed by the snapshot | Prevents selectively including early completions from immature cohorts |
| Known late outcome | Delivered after deadline, or still in transit after deadline | An open order can already be known to have failed the promise |
| Unknown outcome | Delivered status but missing/invalid delivery timestamp | Unknown must not automatically become success or failure |
| Lead-time cohort | Completed orders with valid delivery timestamps | A different population from the service-rate cohort |

The three main tables in memory are:

- `d`: all unique cleaned orders, including open and unknown-outcome orders.
- `eligible`: orders already due with known service outcomes.
- `completed`: orders with valid observed lead times, including any already completed orders not yet due.

**Transferable takeaways**

1. Define what a row represents before counting or averaging it.
2. A KPI requires an eligibility rule as well as a formula.
3. Two valid analyses can use different populations; name those populations explicitly.

## Source data and feature lineage

In this project the inputs are simulated. 
The system names below describe plausible real-world sources, 
not additional integrations implemented by the code.

| Original field or rule | Plausible source | Meaning and necessary checks |
| --- | --- | --- |
| `order_id` | ERP/WMS order master | Stable order identifier; unique at the chosen grain |
| `released_at` | WMS release event | Defined start of the measured process; valid timezone |
| `delivered_at` | WMS/carrier proof of delivery | Defined finish event; distinguish missing capture from unfinished work |
| `status` | WMS/order tracking | State as of the snapshot, using a consistent status vocabulary |
| `carrier`, `lane` | Shipment allocation/master data | Assignment and comparison strata; normalize labels |
| `distance_km` | Routing/WMS data | Positive distance with consistent planned-versus-actual definition |
| `units` | Order lines aggregated to order | Ordered quantity; not delivered quantity or capacity usage |
| `sla_h` by `lane` | Contract/reference table | Commitment valid for the order; real contracts may need effective dates |
| `SNAPSHOT` | Extraction metadata | Common observation cutoff for all records |

| Derived field | Inputs | Why create it? |
| --- | --- | --- |
| `due_at` | `released_at`, `sla_h` | Turn a service promise into a timestamp against which performance can be judged |
| `lead_h` | `delivered_at`, `released_at` | Express elapsed duration in an operationally meaningful unit |
| `matured` | `due_at`, `SNAPSHOT` | Identify orders for which the service deadline has passed |
| `on_time`, `late` | Deadline, delivery time, status, maturity | Create a nullable outcome without inventing missing results |
| `day` | `released_at` | Form release cohorts and identify shared daily conditions |
| `weekday`, `day_index` | `day` | Describe weekly patterns and elapsed calendar time |
| `daily_orders` | All unique orders grouped by `day` | Approximate released workload; this is not a capacity-utilization measure |
| `distance_missing` | Missingness of `distance_km` | Make unavailable predictor data visible |
| `invalid_delivery_time` | Delivery time, release time, snapshot | Preserve the reason an outcome became unavailable |
| `iqr_flag` on `completed` | Lead times and within-lane quartiles | Prioritize unusual records for review without removing valid delays |

**Transferable takeaways**

1. Each feature should have a business interpretation and a traceable derivation.
2. A convenient proxy such as order count is not interchangeable with the concept it approximates, 
such as workload or utilization.
3. Keep quality flags beside cleaned values so another analyst can reconstruct the reasoning.

## Question 1 — “Can we trust the service figure enough to act on it?”

**Decision.** Determine whether apparent poor performance warrants operational intervention, data repair, or both.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| KPI | Exact-duplicate count, invalid-time count, outcome coverage, and missing-outcome sensitivity bounds |
| Best display | A compact reconciliation table for exact counts; a bar chart of missingness by source/group when diagnosing where the problem occurs |
| Derived information | Unique-order population, validity flags, maturity, known-versus-unknown outcome |
| Original data | IDs, timestamps, status, lane commitments, snapshot |
| Method/tools | `pandas.duplicated`, `drop_duplicates`, `isna`, Boolean checks, validated `merge` |

Start with a reconciliation: 
6,054 raw rows become 6,024 unique orders after removing 30 exact copies. 
The pipeline flags 12 impossible delivery times. 
Of 5,992 orders already due, 5,955 have known outcomes and 37 have unknown outcomes. 
The remaining 32 orders are not yet due.

Do not use `drop_duplicates("order_id")` to silently pick one conflicting version of an order. 
The code removes exact copies, then asserts key uniqueness. 
Likewise, it uses a many-to-one join to prevent duplicate reference keys from multiplying orders.

Implemented in pipeline sections 3–5. 
Outputs: `raw_missingness.csv`, `quality_audit.csv`, and `metrics.json`.

```python
# After section 4: coverage of matured outcomes.
coverage = len(eligible) / d["matured"].sum()

# After section 5: extreme assumptions about unavailable outcomes.
successes = int(eligible["on_time"].sum())
unknown = int((d["matured"] & d["on_time"].isna()).sum())
denominator = int(d["matured"].sum())
lower = successes / denominator
upper = (successes + unknown) / denominator
```

The observed known-outcome on-time rate is 43.85%. 
Treating every unknown matured outcome as a failure or as a success gives 43.57%–44.19%. 
This bounds the effect of those missing outcomes; 
it does not account for incorrect recorded statuses, missing orders, or selection in the source extract.

**Transferable takeaways**

1. Quantify how data quality could change the decision, rather than reporting missingness alone.
2. Preserve reconciliation counts from raw data to the final denominator.
3. Sensitivity bounds address unavailable information; they are not statistical confidence intervals.

## Question 2 — “Are we meeting our delivery promises?”

**Decision.** Assess service reliability for the eligible order population and identify lanes needing attention.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| KPI | On-time orders divided by known outcomes among orders already due |
| Best display | KPI plus numerator/denominator for the overall result; grouped bars or dots for carrier-by-lane comparisons |
| Derived information | `due_at`, `matured`, `on_time`, `late` |
| Original data | Release time, delivery time, status, lane, SLA table, snapshot |
| Method/tools | `pd.to_timedelta`, timestamp comparisons, Boolean values, `groupby().agg()` |

Duration alone cannot answer this question. 
A 30-hour delivery is late against a 24-hour promise and on time against a 48-hour promise. 
In this case, Local uses 24 hours and Regional uses 48 hours.

```python
# Core feature logic from section 4.
d["due_at"] = d["released_at"] + pd.to_timedelta(d["sla_h"], unit="h")
d["matured"] = d["due_at"] <= SNAPSHOT

# Section 5 operates on the already constructed eligible population.
service = eligible.groupby(["carrier", "lane"]).agg(
    orders=("order_id", "size"),
    successes=("on_time", "sum"),
)
service["rate"] = service["successes"] / service["orders"]
```

A still-open order past its deadline is already late. 
A not-yet-due order is excluded even if it finished early, 
to avoid selecting only successful early completions from recent cohorts. 
A delivered order with an unavailable timestamp remains unknown.

Implemented in sections 4–5 and 10. 
Outputs: `service_by_lane.csv`, `overview.png`. 

**Transferable takeaways**

1. Convert a business promise into an explicit evaluable rule.
2. Distinguish incomplete observation from failure and from missing measurement.
3. Pair percentages with counts; a rate without its denominator hides exposure.

## Question 3 — “What delivery experience is typical, and how bad is the tail?”

**Decision.** Understand consistency, customer experience, and whether a service commitment is plausible.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| Metrics | Median, P90, mean, standard deviation, sample size |
| Best display | ECDF for “what fraction finishes by X hours?”; within-lane box plots for compact carrier comparisons |
| Derived information | `lead_h`; grouping by lane and carrier |
| Original data | Valid release and delivery timestamps plus group identifiers |
| Method/tools | Datetime subtraction, `agg`, `quantile`, Seaborn `ecdfplot` and `boxplot` |

The median describes the center without being dominated by a few long delays. 
P90 asks about the slower part of the observed distribution: 
approximately 90% of measured durations are at or below that value. 
The mean remains useful for some aggregate planning questions, 
but one mean cannot describe both the center and tail.

An ECDF uses hours on the horizontal axis and cumulative share on the vertical axis. 
Read upward from a proposed time commitment to see the observed completion share, 
or sideways from 0.90 to locate P90. 
Separate lanes because a mixture of short and long routes can hide the pattern of either.

```python
# Section 5, using the completed-order population.
duration_summary = completed.groupby(["carrier", "lane"])["lead_h"].agg(
    n="size", median_h="median", mean_h="mean",
    p90_h=lambda s: s.quantile(.90), sd_h="std",
)

# Section 10.
sns.ecdfplot(data=completed, x="lead_h", hue="lane")
```

For Local orders, the supplied output gives A a median of 20.34 hours and P90 of 25.25; 
B has 23.26 and 28.19 hours. 
These are completed-order summaries, not unbiased estimates of all orders' eventual durations.

Implemented in sections 5 and 10. 
Outputs: `lead_time_summary.csv`, `overview.png`. 
The box plot hides individual outlier markers for readability but retains those observations in calculations; 
the ECDF retains the observed tail.

**Transferable takeaways**

1. Match the statistic to the experience being discussed: typical, variable, or extreme.
2. Use distribution plots when a single average would conceal business risk.

## Question 4 — “Which carrier performs better on comparable work?”

**Decision.** Inform a carrier review without confusing assignment mix with performance.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| KPI | Within-lane on-time rates and rates standardized to one common lane mix |
| Best display | Side-by-side raw and standardized bars; a carrier-by-lane table showing rates and counts |
| Derived information | Group rates, common lane weights, weighted sum per carrier |
| Original data | Carrier assignment, lane, and the valid service outcome cohort |
| Method/tools | `groupby`, `unstack`, `value_counts(normalize=True)`, aligned weighted arithmetic |

Carrier A receives more Regional work, where the observed service rate is lower. 
A raw average therefore combines carrier performance and allocation policy. 
Ask first whether both carriers appear in each lane; 
without overlap, a within-lane comparison cannot be learned from these data alone.

Use the same lane weights for both carriers:

`standardized_rate(carrier) = sum(common_lane_weight × carrier_rate_in_lane)`

```python
# Section 8.
within = eligible.groupby(["carrier", "lane"])["on_time"].mean().unstack()
weights = eligible["lane"].value_counts(normalize=True).reindex(within.columns)
assert within.notna().all().all()
standardized = within.mul(weights, axis=1).sum(axis=1)
```

| Carrier | Raw on-time rate | Common-lane-mix rate |
| --- | ---: | ---: |
| A | 36.48% | 56.22% |
| B | 51.05% | 34.76% |

**This reversal is the central case finding**. 
It justifies examining allocation before judging the carriers. 
It does not prove that switching an order from B to A would produce the standardized difference: 
within-lane orders may still differ, and a large reallocation could alter carrier capacity and behavior.

Implemented in sections 8 and 10. 
Outputs: `carrier_comparison.csv`, `service_by_lane.csv`, `overview.png`.

**Transferable takeaways**

1. Compare aggregate results with results inside meaningful strata (Simpson's Paradox).
2. Choose and disclose reference weights; an adjusted average always represents some population.

## Question 5 — “Is performance changing over time?”

**Decision.** Identify when to investigate changing service, 
while avoiding reactions to small-denominator noise.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| Metrics | Daily success and eligible counts, daily rate, seven-day pooled rate |
| Best display | Time-ordered line chart with raw daily values and a clearly labeled smoother; companion volume information |
| Derived information | Release `day`, daily counts, rolling sums |
| Original data | Release timestamp and eligible service outcomes |
| Method/tools | `groupby`, `reindex`, `rolling`, Matplotlib date-axis formatting |

Grouping by release date answers *“how did the cohort entering the process that day perform?”* 
Grouping by delivery date would instead describe completions occurring that day. 
Neither is universally correct; they answer different questions.

Compute a seven-day service rate from the pooled numerator and denominator. 
Averaging seven percentages gives a low-volume day the same weight as a high-volume day.

```python
# Section 6, after daily counts have been reindexed to a daily calendar.
daily["rate"] = daily["success"] / daily["n"]
daily["rolling_7d_rate"] = (
    daily["success"].rolling(7, min_periods=7).sum()
    / daily["n"].rolling(7, min_periods=7).sum()
)
```

Implemented in sections 6 and 10. 
Outputs: `daily_service.csv`, `overview.png`. 
The dashboard shows daily counts in hover information. 
A rolling line is not a control limit or a forecast.

**Transferable takeaways**

1. Pick the date that answers your question: 
release date shows how orders started that day did; 
delivery date shows what was completed that day.
2. For a seven-day rate, add up all successful orders and all eligible orders 
across those days, then divide. 
Do not average the seven daily percentages.
3. A smoothed line makes the overall pattern easier to see, 
but it does not prove that service has really changed. 
Check the daily counts and investigate before acting.

## Question 6 — “What is associated with longer lead times?”

**Decision.** Form hypotheses about route difficulty and workload that warrant operational investigation.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| Metrics | Pearson and Spearman association; within-lane association |
| Best display | Scatter plot colored by lane |
| Derived information | `lead_h`, `daily_orders`, `weekday`, `day_index`; categorical carrier and lane |
| Original data | Timestamps, distance, order quantity, assignment, all released orders for workload counts |
| Method/tools | SciPy `pearsonr`/`spearmanr`, grouped Pandas correlation, Seaborn scatter plots |

The supplied pooled distance–lead-time Pearson correlation is about 0.94. 
Much of that pattern separates Local from Regional orders. 
A plot reveals these groups; a single correlation coefficient conceals them. 
Examine within-lane patterns before treating distance as a complete explanation.

```python
# Section 8: retain matched pairs when handling missing data.
pair = completed[["distance_km", "lead_h"]].dropna()
r = stats.pearsonr(pair["distance_km"], pair["lead_h"]).statistic
within_lane = completed.groupby("lane")[["distance_km", "lead_h"]].corr(
    method="spearman"
)
```

`daily_orders` attaches the same total release count to all orders from a day. 
Use `groupby(...).transform("size")` to preserve order-level rows. 

Outputs: `within_lane_correlations.csv`, `relationship.png`. 
The plotted scatter sample is capped at 1,500 observations for readability; 
the correlations use all available valid pairs.

**Transferable takeaways**

1. Look at the group structure before interpreting a coefficient.
2. Explanatory features need operational meaning, not merely predictive association.
3. A feature valid for retrospective analysis may be unavailable at prediction time.

## Question 7 — “How stable is the estimate, and which extreme records should we review?”

**Decision.** Distinguish uncertainty about the process from data defects and genuine operational disruptions.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| Metrics | Day-bootstrap interval for overall service; within-lane IQR flags for unusual durations |
| Best display | Distribution plot plus a review table for flagged records |
| Derived information | Daily success/eligible counts; lane quartiles and IQR |
| Original data | Outcome, release day, valid duration, lane |
| Method/tools | NumPy random resampling and quantiles; Pandas grouped `transform` and Boolean flags |

These are two different tasks. 
An uncertainty interval asks how a process estimate might vary under repeated sampling assumptions. 
An outlier flag asks which observations deserve individual review.

Section 7 uses transparent within-lane screening:

```python
q1 = completed.groupby("lane")["lead_h"].transform(lambda s: s.quantile(.25))
q3 = completed.groupby("lane")["lead_h"].transform(lambda s: s.quantile(.75))
iqr = q3 - q1
flag = (completed["lead_h"] < q1 - 1.5 * iqr) | (
    completed["lead_h"] > q3 + 1.5 * iqr
)
review = completed.loc[flag]
```

An impossible negative duration indicates invalid data. 
An exceptionally long positive duration may record a genuine disruption. 
The pipeline keeps valid extremes in its statistics and exports `lead_time_flags.csv` for review. 
A flag is neither a root cause nor a calibrated probability of error.

**Transferable takeaways**

1. State what an interval refers to: a mean, a rate, a contrast, or a future observation.
2. Investigate extremes before deleting them; the tail may contain the business problem.

## Question 9 — “What should management monitor, and what action is justified?”

**Decision.** Convert analysis into an accountable review process and a testable next step.

**Reasoning chain**

| Link | Design choice and reason |
| --- | --- |
| Metrics | Eligible volume, on-time rate, unknown-outcome count, distribution and trend views |
| Best display | A small dashboard with consistent filters and visible denominators; a written decision note for recommendations |
| Derived information | Filtered cohorts and recomputed numerators/denominators |
| Original data | Clean order dataset, snapshot, filter definitions, KPI contract |
| Method/tools | Streamlit widgets and metrics, Plotly line/ECDF charts, Pandas aggregation, CSV/HTML exports |

The dashboard lets a manager select release dates, carriers, and lanes. 
It shows total selected orders, known matured outcomes, on-time rate, 
unknown matured outcomes, daily service, completed-order distributions, 
and a downloadable table.
When a filter changes, the population changes. 

Implemented in `app.py` and section 11. 
Read the generated `report.html` alongside the underlying tables. 
Optional Plotly support embeds an interactive relationship chart in the report; 
Streamlit requires a running Python process.

A defensible management note for this simulation is:

> Known matured orders have a 43.85% on-time rate. 
> Carrier B leads in the raw aggregate, 
> while A leads after comparison at a common lane mix. 
> Review assignment policy and within-lane performance 
> before changing the carrier contract. 
> Audit missing outcomes and investigate the largest late-order groups. 

Do not infer that replacing B with A will mechanically deliver the standardized uplift. 
That requires evidence about comparable assignments, available capacity, intervention effects, and cost.

**Transferable takeaways**

1. A dashboard should preserve the analytical contract while allowing exploration.
2. Separate an observed pattern, its possible explanation, and the proposed action.
3. Attach a next investigation or experiment to every consequential recommendation.

## Questions the current data cannot answer

Recognizing missing evidence is part of analytical competence. 
These are extensions, not existing pipeline results.

| New management question | KPI/feature needed | Additional original data | Appropriate first display and tools |
| --- | --- | --- | --- |
| “Is the delay inside the warehouse or in transport?” | Waiting, picking, staging, and transit durations | Order-level event milestones with consistent event definitions | Stacked component summaries and component distributions; Pandas event joins and timestamp subtraction |
| “Are orders delivered on time and in full?” | OTIF outcome under a documented order-level rule | Ordered and delivered quantities, partial deliveries, promised date | OTIF rate by segment with counts; Pandas line-to-order aggregation |
| “Which failures cost us most?” | Observed penalty/cost per order; total cost by cause | Actual penalties, service costs, customer/order linkage, verified cause codes | Sorted cost bars/Pareto; Pandas validated joins and aggregation |
| “Would extra staffing improve service?” | Intervention effect and incremental cost | Staffing, capacity, backlog, task complexity, intervention timing/assignment | Outcome trends and a planned comparison design; statistical modeling after design checks |
| “What are eventual lead times including open orders?” | Time-to-delivery distribution with censoring | Entry time, observation cutoff, reliable delivery/censoring indicator | Survival curve; a survival-analysis package after checking censoring assumptions |

Do not infer causal warehouse delays from release-to-delivery duration alone. Do not treat ordered units as evidence that the customer received them. Do not convert association into a savings estimate without an intervention model.

## A reusable question-design worksheet

Before adding another chart or feature, complete the following for any domain:

| Step | Student prompt | Expected deliverable |
| --- | --- | --- |
| 1. Decision | Who will act, and what could change? | One concrete decision sentence |
| 2. Question | Which uncertainty prevents that decision? | A comparison, threshold, trend, distribution, or causal question |
| 3. KPI | What quantity answers that question? | Formula, units, grain, numerator, denominator, time window |
| 4. Display | What should the reader compare visually? | Chart choice with axes, groups, sample sizes, and uncertainty specified |
| 5. Features | Which concepts are not directly recorded? | Derived-field definitions with business meaning |
| 6. Evidence | Which original observations and rules are required? | Source fields, keys, timestamps, business rules, provenance |
| 7. Computation | What is the simplest adequate implementation? | Pandas aggregation, statistical method, and plotting API |
| 8. Challenge | What could reverse or invalidate the result? | Missingness, selection, dependence, confounding, sensitivity checks |
| 9. Action | What conclusion is justified now? | Observation, limitation, proposed test, owner and follow-up measure |

The generic lessons at each step are: 
1. define a decision before a metric; 
2. define a metric before a plot; 
3. derive only interpretable features; 
4. request evidence before claiming an answer; 
5. and make the assumptions visible before recommending action.

## Computation choices and official references

| Need | First tool to reach for | Why it fits this project |
| --- | --- | --- |
| Read, validate, join, reshape, aggregate | [Pandas](https://pandas.pydata.org/docs/user_guide/index.html) | Preserves labeled columns, group keys, and explicit denominators |
| Missing outcomes | [Pandas missing-data guide](https://pandas.pydata.org/docs/user_guide/missing_data.html) | Supports explicit missingness rather than invented values |
| Safe integration | [Pandas merge guide](https://pandas.pydata.org/docs/user_guide/merging.html) | Validates cardinality and reveals unmatched keys |
| Group statistics | [Pandas groupby guide](https://pandas.pydata.org/docs/user_guide/groupby.html) | Computes one transparent summary per comparison group |
| Simulate and resample | [NumPy random generator](https://numpy.org/doc/stable/reference/random/generator.html) | Reproducible random draws and vectorized bootstrap arithmetic |
| Correlation and statistical functions | [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html) | Standard estimators, with assumptions still chosen by the analyst |
| Adjusted associations | [statsmodels formula examples](https://www.statsmodels.org/stable/example_formulas.html) | Readable formulas and model summaries |
| Dependence-aware model uncertainty | [statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html) | Supports cluster covariance when the dependence structure warrants it |
| Static distributions and comparisons | [Seaborn tutorials](https://seaborn.pydata.org/tutorial.html) | Concise plots from tidy, labeled data |
| Axes, labels, export | [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html) | Precise control for reports |
| Interactive charts | [Plotly Express](https://plotly.com/python/plotly-express/) | Hover details, facets, and browser interaction |
| Dashboard controls | [Streamlit documentation](https://docs.streamlit.io/) | Filters, metrics, tables, and downloads in Python |
