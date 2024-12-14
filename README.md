# BV Pricing Framework

A framework to recommend and execute prices. The framework consists of multiple actions that includes dynamic pricing, change in base prices, short-stay configuration and more. In other terms, the framework can interact with LARS and update prices and configurations.

Note: The framework is used in multiple other repositories, because it holds the LARS engine. However, the LARS engine is currently being built in a separate repository, which will lead to that the requirements of bv_pricing will be changed to lars_engine in the future.

## QUICKSTART Again

### Installing (production)

Run the following commands to create a separate directory and virtual enviroment from which to run the bv-pricing framework.

```bash
git clone git@bitbucket.org:ovhpricing/bv_pricing.git
make prod-env
```

Alternatively run these commands

### Development

Python Dependencies using pip and virtualenv

```console
git clone git@bitbucket.org:ovhpricing/bv_pricing.git
make dev-env
```

To install the repository inside another repository use the followig command

```bash
pip install git+ssh://git@bitbucket.org/ovhpricing/bv_pricing.git@22.06.03
```

### Enviroment variables

Run the following command to add the two lines to the end of the `env/bin/activate` file. These lines will ensure that when activating the virtual enviroment via `source env/bin/activate` then the enviroment variables is loaded.

```bash
printf "\n# Adding this command to read local .env file" >> env/bin/activate
printf "\nexport \$(grep -v '^#' .env | xargs)" >> env/bin/activate
```

Create a `.env` file based on the template `dotenv`.

### Tagging

Stage all changes
git commit
git push
git tag 22.05.19.rc3
git push origin 22.05.19.rc3

## Usage

To use the framework, it muse be installed within a virtual environment. You can now view the options the framework offers by the following command

```bash
bv-pricing --help
```

To run a specific command

```bash
bv-pricing [COMMAND]
```
