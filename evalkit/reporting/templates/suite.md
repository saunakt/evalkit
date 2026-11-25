# {{ r.suite }} — {{ r.status }}

Run `{{ r.run_id }}`{% if r.doc %} · {{ r.doc }}{% endif %}


## Verdict

**{{ r.status }}**
{% for f in r.failures %}
- FAIL: {{ f }}
{% endfor %}
{% for w in r.warnings %}
- WARN: {{ w }}
{% endfor %}
{% if not r.failures and not r.warnings %}
All checks passed.
{% endif %}

## What was tested

- Cases: {{ r.tested.cases }} · repeats: {{ r.tested.repeats }} · seed: {{ r.tested.seed }}
- Target: {{ r.tested.target }} · profile: {{ r.tested.profile | fmt }}
- Metrics: {% for x in r.metrics %}`{{ x.metric }}`{% if not loop.last %}, {% endif %}{% endfor %}


## Results vs thresholds

Overall case pass rate {{ r.overall.pass_rate | pct }} (95% CI {{ r.overall.lo | pct }}–{{ r.overall.hi | pct }}, n={{ r.overall.n }}).

| metric | value | pass rate | 95% CI | n | target | status |
|---|---|---|---|---|---|---|
{% for x in r.metrics %}
| `{{ x.metric }}` | {{ x.value | fmt }} | {{ x.pass_rate | pct }} | {% if x.lo is not none %}{{ x.lo | pct }}–{{ x.hi | pct }}{% else %}–{% endif %} | {{ x.n }} | {{ x.target | fmt }} | {{ x.status }} |
{% endfor %}

## Comparison with baseline

{% if r.comparison.baseline %}
Baseline `{{ r.comparison.baseline }}`.

| metric | baseline | current | Δ | p | fixed | newly broken |
|---|---|---|---|---|---|---|
{% for metric, d in r.comparison.metrics.items() %}
| `{{ metric }}` | {{ d.base_rate | pct }} | {{ d.cand_rate | pct }} | {{ d.delta | fmt }} | {{ d.p_mcnemar | fmt }} | {{ d.fixed | join(", ") }} | {{ d.broken | join(", ") }} |
{% endfor %}
{% else %}
No baseline run.
{% endif %}

## Results by slice

{% for by, rows in r.slices.items() %}{% if rows %}
| {{ by }} | n | pass rate |
|---|---|---|
{% for s in rows %}
| {{ s.slice }} | {{ s.n }} | {{ s.pass_rate | pct }} |
{% endfor %}

{% endif %}{% endfor %}
## Failing and flaky cases

{% for f in r.failing %}
- `{{ f.case_id }}`{% if f.case_id in r.flaky %} (flaky){% endif %} — pass rate {{ f.pass_rate | pct }}{% if f.must_pass %}, must-pass{% endif %}: {{ f.failed | join(", ") }}
{% else %}
No failing cases.
{% endfor %}

## Suite panels

{% for p in r.panels %}
### {{ p.title }}

{% if p.kind == "table" %}
| {{ p.columns | join(" | ") }} |
|{% for c in p.columns %}---|{% endfor %}

{% for row in p.rows %}
| {% for v in row %}{{ v | fmt }}{% if not loop.last %} | {% endif %}{% endfor %} |
{% endfor %}
{% elif p.kind == "bars" %}
{% for it in p["items"] %}
- {{ it.label }}: {{ it.value | fmt }}{% if it.n is defined and it.n is not none %} (n={{ it.n }}){% endif %}

{% endfor %}
{% elif p.kind == "kv" %}
{% for k, v in p["items"].items() %}
- {{ k }}: {{ v | fmt }}
{% endfor %}
{% else %}
{{ p.text }}
{% endif %}

{% else %}
No panels.
{% endfor %}

## Judge validity

{% for j in r.judges %}
- `{{ j.id }}` v{{ j.version }} ({{ j.provider }}:{{ j.model }}): {{ "calibrated" if j.calibrated else "UNCALIBRATED" }} — {{ j.reason }}
{% else %}
No judges used.
{% endfor %}

## Not tested

{% for n in r.not_tested %}
- `{{ n.metric }}`: {{ n.reason }}
{% else %}
Every selected metric was computed.
{% endfor %}
