"""Explicit runtime helpers for recovered workflows; no historical approvals invented."""
from pathlib import Path
import csv
import os
import datetime


def display(value):
    print(value.to_string() if hasattr(value, 'to_string') else value)


def log_decision(step, decision, reason, rows=None, hours=None):
    path = Path(os.environ['BTP_ROOT']) / 'cleaning/decision_log_A.csv'
    columns = ['date', 'step', 'decision', 'reason', 'rows_affected', 'hours_affected',
               'event_origin', 'recovery_run_id', 'linguistic_approval']
    previous = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as f:
            previous = list(csv.DictReader(f))
    if any(row['decision'] == decision for row in previous):
        return
    date = os.environ.get('NYISHI_RECOVERY_UTC', datetime.datetime.now(datetime.timezone.utc).isoformat())[:10]
    with path.open('a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not previous:
            writer.writeheader()
        writer.writerow(dict(date=date, step=step, decision=decision, reason=reason,
                             rows_affected=rows, hours_affected=hours,
                             event_origin='historical_recipe_reproduced_now',
                             recovery_run_id=os.environ['NYISHI_RUN_ID'],
                             linguistic_approval='not_established; recovery reproduction authorized'))
