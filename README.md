# Stock Market Price Detection


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
1. python3
2. pip3
3. virtualenvwrapper

Install `python3`
```bash
sudo apt-get install -y python3
```
Install `pip3`
```bash
sudo apt-get install -y python3-pip
```

Install `virtualenvwrapper`
```bash
sudo pip3 install vitualenvwrapper
```

On your `.zshrc` or `.bashrc`
```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/projects
source /usr/local/bin/virtualenvwrapper.sh
```

```bash
source ~/.bashrc
#or
source ~/.zshrc
```

### Installing

Create a virtualenv

```bash
mkvirtualenv awesome-python
workon awesome-python
```

Clone the Repo

```bash
export PROJECT_NAME=stock_market_prediction
git clone `clone url` $PROJECT_NAME
cd $PROJECT_NAME
```

Install the requirements

```bash
pip install -r requirements.txt
```

## Running the tests

```bash
python -m unittest discover -v
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [unittest](https://docs.python.org/3/library/unittest.html) - builtins `unittest` framework is used.

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* Tahseen Junayed.

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* Hat tip to anyone whose code was used
