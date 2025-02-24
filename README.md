# Generic AI Sample Applications

This repository contains a list of sample applications that demonstrate a set of common AI use cases.

Namely those are interactions with LLMs via:
- Chat
- Vision
- Audio

# Setup your environment
To make the setup easier, this repository contains configurations and infrastructure as code templates to setup the required cloud services and provide the neccessary environment to run the provided samples.

To setup everything, run the following command from the Azure Developer CLI (azd). To get more information about azd, you can check out the documentation here (https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/)

```
echo "login into your azure environment"
azd auth login

echo "deploy the infrastructure. Follow the instructions after you run the command"
azd up
```

If you cannot run the azd command, check out the .env_sample file and fill in the missing variables by hand.

The above process will generate a .env file in your root directory. This .env file will contain all required connections information you need to run the samples applications.

# Run the samples
The repository contains several projects. To run a specific sample, cd into the folder of the sample, run  `pip install - r requirements.txt` and execute the run command.

For **chat**, **ingestions-pipeline**, **voice-interaction** run

`python -m app`

If you want to debug the sample, you can alternatively run the app via the debugger in VSCode. You might need to change the launch config and provide the correct module to start.

For **chat_langgraph** you need to use `langgraph dev --debug-port 2026` as this command spuns up a developer UI that helps you debug your agents. This will automatically start the debugger.

