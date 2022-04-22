VENV = venv

# If `venv/bin/python` exists, it is used. If not, use PATH to find python.
SYSTEM_PYTHON  = $(shell which python3.9)	# Make sure that you have python3.9 in your system!!
PYTHON         = $(or $(wildcard venv/bin/python), $(SYSTEM_PYTHON))

YELLOW=\033[1;33m
GREEN=\033[0;32m
NC=\033[0m# No Color

install: venv deps
	@echo "${GREEN}Hopefully no error ocurred.${NC}" 
	@echo "Don't forget to turn on the virtual environment using the following command:"
	@echo "${YELLOW}source venv/bin/activate${NC}"

venv:
	rm -rf $(VENV)
	$(SYSTEM_PYTHON) -m venv $(VENV)

deps: venv requirements.txt
	$(PYTHON) -m pip install -r requirements.txt