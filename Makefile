install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest -vv test_model.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	
	echo "\n## Feature Importance Plot" >> report.md
	echo "![Feature Importance](./Results/feature_importance.png)" >> report.md
	
	cml comment create report.md
		
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add Results/metrics.txt Results/previous_metrics.txt
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git fetch origin
	git pull origin update --rebase || git pull origin update --no-rebase
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload nglinhbao/Housing-Pricing ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload nglinhbao/Housing-Pricing ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload nglinhbao/Housing-Pricing ./Results /Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

validate:
	python model_validation.py

all: install lint test train validate eval