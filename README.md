```bash
git clone https://github.com/GongGuangyu/PlanGuard.git
cd PlanGuard
pip install -r requirements.txt
cp .env
API_KEY=sk-your-real-key-here
python eval_all_usertool_dh.py    # for all tools
python eval_one_usertool_dh.py    # for one tool
```


PlanGuard/
├── agent.py              # Main agent
├── planner.py            # Planning module
├── llm_guard.py          # LLM guardrail
├── injec_dh_tools.py     # Injection detection tools
├── requirements.txt      # Dependencies
├── .env.example          # API key template
├── user_cases.jsonl      # User test cases
└── attacker_cases_dh.jsonl  # Attacker test cases