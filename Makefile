.PHONY: tests help init_env init_git pre-commit_update docs_view docs_test test check

####----Basic configurations----####

install_env: ## Install libs with UV and pre-commit
	@echo "🚀 Creating virtual environment using UV"
	uv sync --all-groups
	@echo "🚀 Installing pre-commit..."
	uv run pre-commit install
	@echo "💻 Activate virtual environment..."
	@bash -c "source .venv/bin/activate"

init_git: ## Initialize git repository
	@echo "🚀 Initializing local git repository..."
	git init -b main
	git add .
	git commit -m "🎉 Initial commit"
	@echo "🚀 Local Git already set!"

####----Install Libraries----####

install_data_libs: ## Install pandas, scikit-learn, Jupyter, seaborn
	@echo "🚀 Installing data science libraries..."
	uv add "pandas[parquet]" numpy scipy scikit-learn
	@echo "🚀 Installing Jupyter, matplotlib and seaborn in dev..."
	uv add jupyter matplotlib seaborn --group dev


####----Tests----####
test: ## Test the code with pytest and coverage
	@echo "🚀 Testing code: Running pytest"
	@uv run pytest --cov

test_verbose: ## Test the code with pytest and coverage in verbose mode
	@echo "🚀 Testing code: Running pytest in verbose mode"
	@uv run pytest --no-header -v --cov

test_coverage: ## Test coverage report coverage.xml
	@echo "🚀 Testing code: Running pytest with coverage"
	@uv run pytest --cov --cov-report xml:coverage.xml

####----Pre-commit----####
pre-commit_update: ## Update pre-commit hooks
	@echo "🚀 Updating pre-commit hooks..."
	uv run pre-commit clean
	uv run pre-commit autoupdate

#

####----Clean----####
clean_env: ## Clean .venv virtual environment
	@echo "🚀 Cleaning the environment..."
	@[ -d .venv ] && rm -rf .venv || echo ".venv directory does not exist"

####----Git----####
switch_main: ## Switch to main branch and pull
	@echo "🚀 Switching to main branch..."
	@git switch main
	@git pull

clean_branchs: ## Clean local branches already merged on the remote
	@echo "🚀 Cleaning up merged branches..."
	@git fetch -p
	@for branch in $$(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v '^\*' | grep -v ' main$$'); do \
		if ! git show-ref --quiet refs/remotes/origin/$$branch; then \
			if git config --get branch.$$branch.remote > /dev/null 2>&1; then \
				echo "Deleting local branch $$branch"; \
				git branch -D $$branch; \
			fi \
		fi \
	done

####----Checks----####
check: ## Run code quality tools with pre-commit hooks.
	@echo "🚀 Linting, formating and Static type checking code: Running pre-commit"
	@uv run pre-commit run -a

lint: ## Run code quality tools with pre-commit hooks.
	@echo "🚀 Linting, formating and Static type checking code: Running pre-commit"
	@uv run pre-commit run ruff

####----Project----####
help:
	@printf "%-30s %s\n" "Target" "Description"
	@printf "%-30s %s\n" "-----------------------" "----------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
