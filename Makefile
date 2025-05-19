run:
	./run.sh

run-10-days:
	PYTHONPATH="." python retell_ai_call_analysis/run.py --days-ago 10

.PHONY: run run-10-days
