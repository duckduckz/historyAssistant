from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.app.conversation_state import update_conversation_state_from_debug


class ConversationStateTests(unittest.TestCase):
    def test_status_ok_execution_updates_last_valid_plan(self) -> None:
        plan = {
            "request_type": "chart",
            "target_tables": ["historyCalls"],
            "metric": "total_calls",
            "date_rule": "last_7_days",
            "filters": [{"field": "shift", "op": "=", "value": "night"}],
            "aggregation": "count",
            "group_by": ["room_name"],
        }
        debug_payload = {
            "question": "Which rooms had repeated calls during the night shift last week?",
            "selected_source": {"source_id": "calls", "source_confidence": 0.65},
            "planning": [
                {
                    "canonical_plan": plan,
                    "validation": {"valid": True, "canonical_plan": plan},
                }
            ],
            "execution": [
                {
                    "execution_result": {
                        "status": "ok",
                        "result_type": "aggregated",
                        "records": [{"room_name": "K 107", "metric_value": 18}],
                    }
                }
            ],
        }

        state = update_conversation_state_from_debug(
            {},
            user_question="Which rooms had repeated calls during the night shift last week?",
            debug_payload=debug_payload,
        )

        self.assertEqual(state["last_source_id"], "calls")
        self.assertEqual(state["last_valid_plan"], plan)
        self.assertTrue(state["last_execution_summary"]["ok"])


if __name__ == "__main__":
    unittest.main()
