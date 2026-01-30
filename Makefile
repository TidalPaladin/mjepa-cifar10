.PHONY: clean clean-env check quality style tag-version test env upload upload-test train train-single

PROJECT=mjepa_cifar10 scripts
QUALITY_DIRS=$(PROJECT)
CLEAN_DIRS=$(PROJECT)
PYTHON=uv run python

# Include training configuration if it exists
-include Makefile.config

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types


clean: ## remove cache files
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-env: ## remove the virtual environment directory
	rm -rf .venv


deploy: ## installs from lockfile
	git submodule update --init --recursive
	which uv || pip install --user uv
	uv sync --frozen --no-dev


init: ## pulls submodules and initializes virtual environment
	git submodule update --init --recursive
	which uv || pip install --user uv
	uv sync --all-groups --all-extras

node_modules: 
ifeq (, $(shell which npm))
	$(error "No npm in $(PATH), please install it to run pyright type checking")
else
	npm install
endif

quality:
	$(MAKE) clean
	$(PYTHON) -m black --check $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a $(QUALITY_DIRS)

style:
	$(PYTHON) -m autoflake -r -i $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a $(QUALITY_DIRS)
	$(PYTHON) -m black $(QUALITY_DIRS)

types: node_modules
	uv run npx --no-install pyright tests $(PROJECT)

update:
	uv sync --all-groups --all-extras

train: Makefile.config ## run distributed training (requires Makefile.config)
	@if [ "$(NUM_TRAINERS)" = "1" ]; then \
		$(MAKE) train-single; \
	else \
		uv run torchrun \
			--standalone \
			--nnodes=1 \
			--nproc_per_node=$(NUM_TRAINERS) \
			scripts/pretrain.py \
			$(CONFIG) \
			$(DATA) \
			--log-dir $(LOG_DIR) \
			-n $(NAME); \
	fi

train-single: Makefile.config ## run single GPU training (requires Makefile.config)
	$(PYTHON) scripts/pretrain.py \
		$(CONFIG) \
		$(DATA) \
		--log-dir $(LOG_DIR) \
		--local-rank $(DEVICE) \
		-n $(NAME)

Makefile.config: ## create Makefile.config from template
	@if [ ! -f Makefile.config ]; then \
		echo "Creating Makefile.config from template..."; \
		cp Makefile.config.template Makefile.config; \
		echo "Please edit Makefile.config to set your training parameters."; \
		exit 1; \
	fi

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
