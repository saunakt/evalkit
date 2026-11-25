# Evaluation summary — {{ r.verdict }}

Run `{{ r.run_id }}` · profile {{ r.info.profile | fmt }} · target {{ r.info.target }}{% if r.baseline %} · baseline `{{ r.baseline }}`{% endif %}


## Gate

**{{ "Gate passed" if r.passed else "Gate FAILED" }}**
{% for s in r.suites %}{% for f in s.failures %}
- FAIL {{ s.suite }}: {{ f }}
{% endfor %}{% endfor %}
{% for f in r.must_pass_failures %}
- must-pass {{ f.suite }}/{{ f.case_id }} failed {{ f.metric }} (repeat {{ f.repeat }})
{% endfor %}

## Suites

| suite | status | cases | pass rate | 95% CI | report |
|---|---|---|---|---|---|
{% for s in r.suites %}
| {{ s.suite }} | {{ s.status }} | {{ s.n }} | {{ s.overall.pass_rate | pct }} | {{ s.overall.lo | pct }}–{{ s.overall.hi | pct }} | {{ s.suite }}/report.md |
{% endfor %}

## Go / no-go checklist

{% for c in r.checklist %}
- {{ c.state }} {{ c.item }}
{% endfor %}

## Model calls and cost by role

| role | calls | paid | free | paid cost (USD) |
|---|---|---|---|---|
{% for role, l in r.ledger.items() %}
| {{ role }} | {{ l.calls }} | {{ l.paid_calls }} | {{ l.free_calls }} | {{ l.paid_cost_usd | fmt(4) }} |
{% endfor %}

## Not tested

{% for n in r.not_tested %}
- {{ n.what }}: {{ n.reason }}
{% else %}
Everything selected was tested.
{% endfor %}
