# Microsoft Certified: Azure Data Scientist Associate

Manage data ingestion and preparation, model training and deployment, and machine learning solution monitoring with Python, Azure Machine Learning and MLflow.

**Overview:**

As a candidate for this certification, you should have subject matter expertise in applying data science and machine learning to implement and run machine learning workloads on Azure. Additionally, you should have knowledge of optimizing language models for AI applications using Azure AI.

Your responsibilities for this role include:

- Designing and creating a suitable working environment for data science workloads.
- Exploring data.
- Training machine learning models.
- Implementing pipelines.
- Running jobs to prepare for production.
- Managing, deploying, and monitoring scalable machine learning solutions.
- Using language models for building AI applications.

As a candidate for this certification, you should have knowledge and experience in data science by using:

- Azure Machine Learning
- MLflow
- Azure AI services, including Azure AI Search
- Azure AI Foundry

## Course: Designing and implementing a data science solution on Azure

### Training: Explore and configure the Azure Machine Learning workspace, Module: Explore Azure ML workspace resources and assets

Throughout this learning path you explore and configure the Azure Machine Learning workspace. Learn how you can create a workspace and what you can do with it. Explore the various developer tools you can use to interact with the workspace. Configure the workspace for machine learning workloads by creating data assets and compute resources.

#### Introduction

Azure Machine Learning provides a comprehensive set of **resources** and **assets** to data scientists to **train**, **deploy**, and **manage** their **machine learning models** on the **Microsoft Azure platform**. To use these resources and assets, you create an **Azure ML workspace** resource in your **Azure subscription**. In the Azure ML workspace, you can **manage data, compute, resources, models, endpoints, and other artifects** related to your ML workloads.

#### Create an Azure ML workspace

Creating an **Azure ML service** in your **Azure subscription** gives access to an **Azure ML workspace**. For reproducibility, the **workspace stores a history of all training jobs, including logs, metrics, outputs, and a snapshot of your code**.

##### Understand the Azure ML service

To create an Azure ML service, you have to:

1. Get access to Azure, for example, through the **Azure portal**.
2. Sign in to get access to an **Azure subscription**.
3. Create a **resource group** with your subscription.
4. Create an **Azure ML service** to create a workspace.

When a workspace is provisioned, Azure automatically creates other Azure resources within the same resource group to support the workspace. 

1. **Azure Storage Account:** To store files and notebooks used in the workspace, and to store metadata of jobs and models.
2. **Azure Key Vault:** To securely manage secrets such as authenticatioon keys and credentials used by the workspace.
3. **Application insights:** To monitor predictive services in the workspace.
4. **Azure Container Registry:** Created when needed to store images for Azure ML environments.

![overview-azure-resources](./notes.assets/overview-azure-resources.png)

##### Create the workspace

You can create an Azure ML workspace in any of the following ways:

- Use the user interface in **Azure portal** to create an Azure ML service.
- Create an **Azure Resource Manager (ARM)** template. [Learn how to use an ARM template to create a workspace](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-workspace-template?tabs=azcli%3Fazure-portal%3Dtrue).
- Use the **Azure CLI** with the Azure ML CLI extension. [Learn how to create the workspace with the CLI v2](https://learn.microsoft.com/en-us/training/modules/create-azure-machine-learning-resources-cli-v2/)
- Use the Azure ML Python SDK.

For example, the following code uses the Python SDK to create a workspace named `mlw-example`: 

```python
from azure.ai.ml.entities import Workspace

workspace_name = "mlw-example"

ws_basic = Workspace(
  name=workspace_name,
  location="eastus",
  display_name="Basic workspace-example",
  description="This example shows how to create a basic workspace",
)
ml_client.workspaces.begin_create(ws_basic)
```

##### Explore the workspce in the Azure portal

Creating an Azure ML workspace typically takes between 5-10 minutes to complete. When your workspace is created, you can select the workspace to view its details.

![workspace-portal](./notes.assets/workspace-portal.png)

From the Overview page of the Azure ML workspace in the Azure portal, you can launch the Azure ML studio. The **Azure ML studio** is a web portal and provides an easy-to-use interface to create, manage, and use resources and assets in the workspace.

From the Azure portal, you can give others access to the Azure ML workspace, using **Access control**.

##### Give access to the Azure ML workspace

You can give individual users or teams access to the Azure ML workspace. Access is granted in Azure using **role-based access control (RBAC)**, which you can configure in the Access control tab of the resource or resource group.

In the access control tab, you can manage permissions to restrict what actions certain users or teams can perform. For example, you could create a policy that only allows users in the *Azure administrators group* to create compute targets and datastores, while users in the *data scientists group* can create and run jobs to train and register models.

There are three general built-in roles that you can use across resources and resource groups to assign permissions to other users:

- **Owner:** Gets full access to all resources and can grant access to others using access control.
- **Contributor:** Gets full access to all resouces but can't grant access to others.
- **Reader:** Can only view the resources but isn't allowed to make any changes.

Additionally, Azure ML has specific built-in roles you can use:

- **AzureML Data Scientist:** Can perform all actions within the workspace, except for creating or deleting compute resources or editing the workspace settings.
- **AzureML Compute Operator:** Is allowed to create, change, and manage access the compute resources within a workspace.

Finally, if the built-in roles aren't meeting your needs, you can create a custom role to assign permissions to other users.

##### Organize your workspaces

Initially, you might only work with one workspace. However, when working on large-scale projects, you might choose to use multiple workspaces. You can use workspaces to group ML assets based on projects, deployment environments (for example, test and production), teams, or some other organizing principle. Learn more about [how to organize Azure Machine Learning workspaces for an enterprise environment](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/ai-machine-learning-resource-organization).

#### Identify Azure ML resources

**Resources** in Azure ML refer to the infrastructure you need to run a ML workflow. Ideally, you want someone like an administrator to create and manage the resources. The resources in Azure ML include:

- The workspace
- Compute resources
- Datastores

##### Create and manage the workspace

The **workspace** is the top-level resource for Azure ML. Data scientists need access to the workspace to train and track models and to deploy the models to endpoints. However, you want to be careful with who has full access to the workspace. Next to references to compute resources and datastores, you can find all logs, metrics, outputs, models, and snapshots of your code in the workspace.

##### Create and manage compute resources

One of the most important resources you need when training or deploying a model is **compute**. There are **five types of compute** in the Azure ML workspace:

- **Compute instances:** Similar to a virtual machine in the cloud, managed by the workspace. Ideal to use as a development environment to run Jupyter notebooks.
- **Compute clusters:** On-demand clusters of CPU or GPU compute nodes in the cloud, managed by the workspace. Ideal to use for production workloads as they automatically scale to your needs.
- **Kubernetes clusters:** Allows you to create or attach an **Azure Kubernetes Service (AKS)** cluster. Ideal to deploy trained machine learning models in production scenarios.
- **Attached computes:** Allows you to attach other Azure compute resources to the workspace, like Azure Databricks or Synapse Spark pools.
- **Serverless compute:** A fully managed, on-demand compute you can use for training jobs. 
  - Note: As Azure ML creates and manages serverless compute for you, it is not listed on the compute page in the studio. Learn more about how to  [use serverless compute for model training](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-serverless-compute).

Though compute is the most important resource when working with machine learning workloads, it can also be the most cost-intensive. Therefore, a best practice is to only allow administrators to create and manage compute resources. Data scientists shouldn't be allowed to edit compute but only use the available compute to run their workloads.

##### Create and manage datastores

The workspace doesn't store any data itself. Instead, all data is stored in **datastores**, which are references to **Azure data services**. The connection information to a data service that a datastore represents, is stored in the **Azure Key Vault**. When a workspace is created, an **Azure Storage account** is created and automatically connected to the workspace. As a result, you have four datastores already added to your workspace:

- `workspaceartifactstore`: Connects to the `azureml` container of the Azure Storage account created with the workspace. Used to store compute and experiment logs when running jobs.
- `workspaceworkingdirectory`: Connects to the file share of the Azure Storage account created with the workspace used by the **Notebooks** section of the studio. Whenever you upload files or folders to access from a compute instance, the files or folders are uploaded to this share.
- `workspaceblobstore`: Connects to the **Blob Storage** of the Azure Storage account created with the workspace. Specifically, the `azureml-blobstore-...` container. Set as the default datastore, which means that whenever you create a data asset and upload data, you store the data in this container.
- `workspacefilestore`: Connects to the file share of the Azure Storage account created with the workspace. Specifically, the `azureml-filestore-...` file share.

Additionally, you can create datastores to connect to other Azure data services. Most commonly, your datastores connects to an Azure Storage Account or **Azure Data Lake Storage (Gen2)** as those data services are most often used in data science projects.

#### Identify Azure ML assets

As a data scientist, you mostly work with assets in the Azure ML workspace. Assets are created and used at various stages of a project and include:

- Models
- Environments
- Data
- Components

##### Create and manage models

The end product of training a model is the model itself. You can train ML models with various frameworks, like Scikit-Learn or PyTorch. A common way to store such models is to package the model as a PyTorch pickle file. Alternatively, you can use the open-source platform MLflow to store your model in the MLModel format (learn more about [logging workflow artifacts as models using MLflow and the MLModel format](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow-models)). Whatever format you choose, binary files represent the model and any corresponding metadata. To persist those files, you can create or register a model in the workspace. When you create a model in the workspace, you specify the name and version. Especially useful when you deploy the registered model, versioning allows you to track the specific model you want to use.

##### Create and manage environments

When you work with cloud compute, it is important to ensure that your code runs on any compute that is available to you. Whether you want to run a script on a compute instance or a compute cluster, the code should execute successfully. 

Imagine working in Python or R and using open-source frameworks to train a model on your device. If you want to use a library such as Scikit-Learn or PyTorch, you have to install it on your device. Similarly, when you write code that uses any frameworks or libraries, you need to ensure the necessary dependencies are installed on the compute that execures the code. To list all necessary requirements, you can create **environments**. When you create an environment, you have to specify the name and version. 

Environments specify software packages, environment variables, and software settings to run scripts. An environment is stored as an image in the Azure Container Registry created with the workspace when it's used for the first time. Whenever you want to run a script, you can specify the environment that needs to be used by the compute target. The environment installs all necessary requirements on the compute before executing the script, making your code robust and reusable across compute targets.

##### Create and manage data

Whereas datastores contain the connection information to Azure data storage services, **data assets** refer to a specific file or folder. You can use data assets to easily access data every time, without having time to provide authentication every time you want to access it. When you create a data asset in the workspace, you specify the path to point to the file or folder and the name and version.

##### Create and manage components

To train ML models, you write code. Across projects, there can be code that you can reuse. Instead of writing code from scratch, you want to reuse snippets of code from other projects. To make it easier to share code, you can create a **component** in a workspace. To create a component, you have to specify the name, version, code, and environment needed to run the code. You can use components when creating **pipelines**. A component therefore often represents a step in a pipeline, for example, to normalize data, to train a regression model, or to test the trained model on a validation dataset.

#### Train models in the workspace

To train models with the Azure ML workspace, you have several options:

- Use Automated ML
- Run a Jupyter notebook
- Run a script as a job

##### Explore algorithms and hyperparameter values with Automated ML

When you have a training dataset and you are tasked with finding the best performing model, you might want to experiment with various algorithms and hyperparameter values. Manually experimenting with different configurations to train a model might take long. Alternatively, you can use Automated ML to speed up the process. Automated ML iterates through algorithms paired with feature selections to find the best performing models for your data.

![automated-machine-learning](./notes.assets/automated-machine-learning.png)

 ##### Run a notebook

When you prefer to develop by running code in notebooks, you can use the built-in notebook feature in the workspace. The **Notebooks** page in the studio allows you to edit and run Jupyter notebooks.

![notebooks](./notes.assets/notebooks.png)

All files you clone or create in the Notebooks section are stored in the file share of the Azure storage account created with the workspace. To run notebooks, you use a compute instance as they are ideal for development and work similar to a virtual machine. You can also choose to edit and run notebooks in VS Code while still using a compute instance to run the notebooks.

##### Run a script as a job

When you want to prepare your code to be production ready, it is better to use scripts. You can easily automate the execution of the script to automate any ML workload. You can run a **script** as a job in Azure ML. When you submit a job to the workspace, all inputs and outputs are stored in the workspace.

![job-overview](./notes.assets/job-overview.png)

There are different types of jobs depending on how you want to execute a workload:

- **Command:** Execute a single script
- **Sweep:** Perform hyperparameter tuning when executing a single script
- **Pipeline:** Run a pipeline consisting of multiple scripts or components 
  - Note: When you submit a pipeline you created with the designer, it will run as a pipeline job. When you submit an Automated Machine Learning experiment, it will also run as a job.

#### Exercise - Explore the workspace

In this exercise, you learn how to: 

- Create an Azure ML workspace
- Explore the Azure ML studio
- Run a training job

[Exercise portal](https://microsoftlearning.github.io/mslearn-azure-ml/Instructions/02-Explore-Azure-Machine-Learning.html):

Azure ML provides a data science platform to train and manage ML models. In this lab, you will create an Azure ML workspace and explore the various ways to work with the workspace. The lab is designed as an introduction of the various core capabilities of Azure ML and the developer tools. If you want to learn about the capabilities in more depth, there are other labs to explore.

##### Before you start

You’ll need an [Azure subscription](https://azure.microsoft.com/free?azure-portal=true) in which you have administrative-level access.

##### Provision an Azure ML workspace

An Azure ML workspace provides a central place for managing all resources and assets you need to train and manage your models. You can provision a workspace using the interactive interface in the Azure portal, or you can use the Azure CLI with the Azure ML extension. In most production scenarios, it's best to automate provisioning with the CLI so that you can incorporate resource deployment into a repeatable development and operations (DevOps) process.

In this exercise, you will use the Azure portal to provision Azure ML to explore all options.

1. Sign into the `https://portal.azure.com`.
2. Create a new Azure ML resource with the following settings:

- **Subscription:** Your Azure subscription
- **Resource group:** `rg-dp100-labs`
- **Workspace name:** `mlw-dp100-labs`
- **Region:** Select the geographical region closest to you
- **Storage account:** Note the default new storage account that will be created for your workspace
- **Key Vault:** Note the default new key vault that will be created for your workspace
- **Application insights:** Note the default new application insights resource that will be created for your workspace
- **Container registry:** None (one will be created automatically the first time you deploy a model to a container)

3. Wait for the workspace and its associated resources to be created. This typically takes around 5 minutes.

Note: When you creata an Azure ML workspace, you can use some advanced options to restrict access through a *private endpoint* and specify custom keys for data encryption. We won't use these options in this exercise, but you should be aware of them.

##### Explore the Azure ML studio

Azure ML studio is a web-based portal through which you can access the Azure ML workspace. You can use the Azure ML studio to manage all assets and resources within your workspace.

1. Go to the resource group named **rg-dp100-labs**.
2. Confirm that the resource group contains your Azure ML workspace, an Application Insights, a Key Vault, and a Storage account.
3. Select your Azure ML workspace.
4. Select **Launch Studio**  from **Overview** page. Another tab will open in your browser to open the Azure ML studio.
5. Close any pop-ups that appear in the studio.
6. Note the different pages shown on the left side of the studio. If only the symbols are visible in the menu, select the ☰ icon to expand the menu and explore the names of the pages.
7. Note the **Authoring** section, which includes **Notebooks, Automated ML**, and **Designer**. There are the three ways you can create your own machine learning models within the Azure ML studio.
8. Note the **Assets** section, which includes **Data, Jobs**, and **Models** among other things. Assets are either consumed or created when training or scoring a model. Assets are used to train, deploy, and manage your models and can be versioned to keep track of your history.
9. Note the **Manage** section, which includes **Compute** among other things. These are infrastructural resources needed to train or deploy a machine learning model.

##### Train a model using AutoML

To explore the use of the assets and resources in the Azure ML workspace, let's try and train a model. A quick way to train and find the best model for a task based on your data is by using the **AutoML** option. 

1. Download the training data that will be used at `https://github.com/MicrosoftLearning/mslearn-azure-ml/raw/refs/heads/main/Labs/02/diabetes-data.zip` and extract the compressed files.
2. Back in the Azure ML studio, select the **AutoML** page from the menu on the left side of the studio.
3. Select **+ New Automated ML job**.
4. In the **Basic settings** step, give a unique name to your training job and experiment or use the default values assigned. Select **Next**.
5. In the **Task type & data** step, select **Classification** as the task type, and select **+ Create** to add your training data.
6. On the **Create data asset page**, in the **Data type** step, give a name to your data asset (e.g. `training-data`) and select **Next**.
7. In the **Destination storage type** step, select **From local files** to upload the training data you downloaded previously. Select **Next** .
8. In the **MLTable selection** step, select **Upload folder** and select the folder you extracted from the compressed file downloaded earlier. Select **Next**.
9. Review the settings for your asset and select **Create**.
10. Back in the **Task type & data** step, select the data you just uploaded and select **Next**.
    - Tip: You may need to select the **Classification** task type again before moving to the next step.
11. In the **Task settings** step, select **Diabetic (Boolean)** as your target column, then open the **View additional configuration settings** option.
12. In the **Additional configuration** pane, change the primary metric to **Accuracy**, then select **Save**.
13. Expand the **Limits** option and set the following properties:
    - **Max trials:** 10
    - **Experiment timeout (minutes)**: 60
    - **Iteration timeout (minutes):** 15
    - **Enable early termination:** Checked

14. For **Test data**, select **Train-test split** and verify that the **Percentage test of data** is 10. Select **Next**.
15. In the **Compute** step, verify that the compute type is **Serverless** and the virtual machine size selected is **Standard_DS3_v2**. Select **Next**.
    - Note: Compute instances and clusters are based on standard Azure virtual machine images. For this exercise, the *Standard_DS3_v2* image is recommended to achieve the optimal balance of cost and performance. If your subscription has a quota that does not include this image, choose an alternative image; but near in mind that a larger image may incur higher cost and a smaller image may not be sufficient to complete the tasks. Alternatively, ask your Azure administrator to extend your quota.
16. Review all your settings and select **Submit training job**.

##### Use jobs to view your history

After submitting the job, you will be redirected to the job's page. Jobs allow you to keep track of the workloads you ran and compare them with each other. Jobs belong to an **experiment**, which allows you to group job runs together.

1. Note that in the **Overview** parameters, you can find the job's status, who created, when it was created, and how long it took to run (among other things).
2. It should take 10-20 minutes for the training job to finish. When it is completed, you can also view the details of each individual component run, including the output. Feel free tomexplore the job page to understand how the models are trained. 
3. Azure ML automatically keeps track of your job's properties. By using jobs, you can easily view your history to understand what you or your colleagues have already done. During experimentation, jobs help keep track of the different models you train to compare and identify the best model. During production, jobs allow you to check whether automated workloads ran as expected.

##### Delete Azure resources

When you finish exploring Azure ML, you should delete the resources you've createxd to avoid unnecessary Azure costs.

1. Close the Azure ML studio tab and return to the Azure portal.
2. In the Azure portal, on the **Home** page, select **Resource groups**.
3. Select the **rg-dp100-labs** resource group.
4. At the top of the **Overview** page for your resource group, select **Delete resource group**.
5. Enter the resource group name to confirm you want to delete it, and select **Delete**.

### Training: Explore and configure the Azure Machine Learning workspace, Module: Explore developer tools for workspace interaction

Azure ML provides data scientists with several resources and assets to build and manage ML models. You can create and manage resources and assets by using various tools that interact with Azure ML workspace. Though you can use any tool to perform the same tasks, each tool provides advantages and disadvantages for specific workloads. You can choose which tool or developer approach best fits your needs.

Learning objectives:

In this module, you will learn how and when to use:

- The Azure ML studio
- The Python Software Development Kit (SDK)
- The Azure CLI

#### Explore the studio

The easiest and most intuitive way to interact with the Azure ML workspace is by using the studio. The Azure ML studio is a web portal, which provides an overview of all resources and assets available in the workspace.

#### Access the studio

After you have created an Azure ML workspace, there are two common ways to access the Azure ML studio:

- Launch the studio from the **Overview** page of the Azure ML workspace resource in Azure portal.
- Navigate to the studio directly by signing in at  [https://ml.azure.com](https://ml.azure.com/) using the credentials associated with your Azure subscription.

When you have opened your workspace in the Azure ML studio, a menu appears in the sidebar.

![studio-home](./notes.assets/studio-home.png)

The menu shows what you can do in the studio:

- **Author**: Create new jobs to train and track a ML model.
- **Assets**: Create and review assets you use when training models.
- **Manage**: Create and manage resources you need to train models.

Though you can use each tool at any time, the studio is ideal for quick experimentation or when you want to explore your past jobs. For example, use the studio if you want to verify that your pipeline ran successfully. Or when a pipeline job has failed, you can use the studio to navigate to the logs and review the error messages. For more repetitive work, or tasks that you'd like to automate, the Azure CLI or Python SDK are better suited as these tools allow you to define your work in code.

#### Explore the Python SDK

Data Scientists can use Azure ML to train, track, and manage ML models. As a data scientist, you will mostly work with the Azure ML workspace for your ML workloads. As most data scientists are familiar with Python, Azure ML offers a SDK so that you can interact with the workspace using Python. The Python SDK for Azure ML is an ideal tool for data scientists that can be used in any Python environment. Whether you normally work with Jupyter notebooks or VS Code, you can install the Python SDK and connect to the workspace.

##### Install the Python SDK

To install the Python SDK within your Python environment, you need Python 3.7 or later. You can install the package with `pip`:

```bash
pip install azure-ai-ml
```

Note: When working with notebooks within the Azure ML studio, the new Python SDK is already installed when using Python 3.10 or later. You can use the Python SDK v2 with earlier versions of Python, but you'll have to install it first.

##### Connect to the workspace

After the Python SDK is installed, you will need to connect to the workspace. By connecting, you are authenticating your environment to interact with the workspace to create and manage assets and resources. To authenticate, you need the values of three necessary parameters: 

- `subscription_id`: Your subscription ID
- `resource_group`: The name of your resource group
- `workspace_name`: The name of your workspace

Next, you can define the authentication by using the following code:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
  DefaultAzureCredential(), subscription_id, resource_group, workspace
)
```

After defining the authentication, you need to call `MLClient` for the environment to connect to the workspace. You will call `MLClient` anytime you want to create or update an asset or resource in the workspace. For example, you will connect to the workspace when you create a new job to train a model.

```python
from azure.ai.ml import command
# configure job
job = command(
  code="./src",
  command="python train.py",
  environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
  compute="aml-cluster",
  experiment_name="train_model"
)

# connect to workspace and submit job
returned_job = ml_client.create_or_update(job)
```

##### Use the reference documentation

To efficiently work with the Python SDK, you will need to use the reference documentation. In the reference documentation, you will find all possible classes, methods, and parameters available within the Python SDK. [The reference documentation on the `MLClient` class](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient) includes the methods you can use to connect and interact with the workspace. Moreover, it also links to the possible operations for the various entities like how to list the existing datastores in your workspace. [The reference documentation also includes a list of the classes for all entities](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities) you can interact with. For example, separate classes exist when you want to create a datastore that links to an Azure Blob Storage, or to an Azure Data Lake Gen 2. By selecting a specific class like `AmlCompute` from the list of entities, you can find [a more detailed page on how to use the class and what parameters it accepts](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.amlcompute).

#### Explore the CLI

In this exercise, you learn how to:

- Create resources with the Azure CLI
- Explore the Azure ML workspace with the studio
- Use the Python SDK to train the model

##### Explore developer tools for workspace interaction

You can use various tools to interact with the Azure ML workspace. Depending on what task you need to perform and your preference for developer tool, you can choose which tool to use when. This lab is designed as an introduction to the developer tools commonly used for workspace interaction. If you want to learn how to use a specific tool in more depth, there are other labs to explore.

###### Before you start

You’ll need an [Azure subscription](https://azure.microsoft.com/free?azure-portal=true) in which you have administrative-level access. The commonly used developer tools for interacting with the Azure ML workspace are:

- **Azure CLI** with the Azure ML extension: Ths command-line approach is ideal for the automation of infrastructure. 
- **Azure ML studio:** Use the user-friendly UI to explore the workspace and all of its capabilities.
- **Python SDK** for Azure ML: Use to submit jobs and manage models from an Jupyter notebook, ideal for data scientists.

You will explore each of these tools for tasks that are commonly done with that tool.

###### Provision the infrastructure with the Azure CLI

For a data scientist to train a ML model with the Azure ML, you will need to set up the necessary infrastructure. You can use the Azure CLI with the Azure ML extension to create an Azure ML worspace and resources like a compute instance. To start, open the Azure Cloud Shell, install the Azure ML extension and clone the Git repo.

1. In a browser, open the Azure portal at `https://portal.azure.com`, signing in with your Microsoft account.

2. Select the [>_] (*Cloud Shell*) button at the top of the page to the right of the search box. This opens a Cloud Shell pane at the bottom of the portal. 

3. Select **Bash** if asked. The first time you open the cloud shell, you will be asked to choose the type of shell you want to use (Bash or PowerShell).

4. Check that the correct subscription is specified and that **No storage account required** is selected. Select **Apply**.

5. Remove any ML CLI extensions (both version 1 and 2) to avoid any conflicts with previous versions with the following command. Use `Shift+Insert` to paste your copied code into the Cloud Shell. Ignore any (error) messages that say that the extensions were not installed.

   ```
   az extension remove -n azure-cli-ml
   az extension remove -n ml
   ```

6. Install the Azure ML (v2) extension with the following command:

   ```
   az extension add -n ml -y
   ```

7. Create a resource group. Choose a location close to you.

   ```
   az group create --name "rg-dp100-labs" --location "eastus"
   ```

8. Create a workspace:

   ```
   az ml workspace create --name "mlw-dp100-labs"
   ```

9. Wait for the workspace and its associated resources to be created. This typically takes around 5 minutes. 
   - Troubleshooting tip: Workspace creation error. If you receive an error when creating a workspace through the CLI, you need to provision the resource manually:
     1. In the Azure portal home page, select **+ Create a resource**.
     2. Search for *machine learning* and then select Azure ML. Select **Create**.
     3. Create a new Azure ML resource with the following settings:
        - Subscription: Your Azure subscription
        - Resource group: rg-dp100-labs
        - Workspace name: mlw-dp-100-labs
        - Region: Select the geographical region closest to you
        - Storage account: Note the default new storage account that will be created for your workspace
        - Key vault: Note the default new key vault that will be created for your workspace
        - Application insights: Note the default new application insights resource that will be created for your workspace
        - Container registry: None (one will be created automatically the first time you deploy a model to a container)
     4. Select **Review + create** and wait for the workspace and its associated resources to be created. This typically takes around 5 minutes. 

###### Create a compute instance with the Azure CLI

Another important part of the infrastructure needed to train a ML model is compute. Though you can train models locally, it's more scalable and cost efficient to use cloud compute. When data scientists are developing a ML model in Azure ML workspace, they want to use a virtual machine on which they can run Jupyter notebooks. For development, a **compute instance** is an ideal fit. After creating an Azure ML workspace, you can also create a compute instance using the Azure CLI. In this exercise, you will create a compute instance with the following settings: 

- **Compute name:** Name of compute instance. Has to be unique and fewer than 24 characters.
- **Virtual machine size:** STANDARD_DS11_V2
- **Compute type** (instance or cluster): ComputeInstance
- **Azure ML workspace name**: mlw-dp100-labs
- **Resource group**: rg-dp100-labs

- Use the following command to create a compute instance in your workspace. If the compute instance name contains "XXXX", replace it with random numbers to create a unique name. If you get an error message that a compute instance with the name already exists, change the name and retry the command. 

  ```
  az ml compute create --name "ciXXXX" --size STANDARD_DS11_V2 --type ComputeInstance -w mlw-dp100-labs -g rg-dp100-labs	
  ```

- Troubleshooting tip: Compute creation error
  - IF you receive an error when creating a compute instance through the CLI, you need to provision the resource manually:
    - In the Azure portal, navigate to the Azure Machine Learning workspace named **mlw-dp100-labs**.
    - Select the Azure Machine Learning workspace, and in its **Overview** page, select **Launch studio**. Another tab will open in your browser to open the Azure Machine Learning studio.
    - Close any pop-ups that appear in the studio.
    - Within the Azure Machine Learning studio, navigate to the **Compute** page and select **+ New** under the **Compute instances** tab.
    - Give the compute instance a unique name and then select **Standard_DS11_v2** as the virtual machine size.
    - Select **Review + create** and then select **Create**.

###### Create a compute cluster with the Azure CLI

Though a compute instance is ideal for development, a compute cluster is better suited when we want to train ML models. Only when a job is submitted to use the compute cluster will resize to more than 0 notes and run the job. Once the compute cluster is no longer needed, it will automatically resize back to 0 nodes to minimize costs.

To create a compute cluster, you can use the Azure CLI similar to creating a compute instance. You will create a compute cluster with the following settings:

- **Compute name:** aml-cluster

- **Virtual machine size:** STANDARD_DS11_V2

- **Compute type:** AmlCompute (Creates a compute cluster)

- **Maximum instances:** Maximum number of nodes

- **Azure ML workspace name:** mlw-dp100-labs

- **Resource group:** rg-dp100-labs

- Use the following command to create a compute cluster in your workspace.

  ```
  az ml compute create --name "aml-cluster" --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute -w mlw-dp100-labs -g rg-dp100-labs
  ```

###### Configure your workstation with the Azure ML studio

Though the Azure CLI is ideal for automation, you may want to review the output of the commands you executed. You can use the Azure ML studio to check whether resources and assets have been created, and to check whether jobs ran successfully or review why a job failed.

1. In the Azure portal, navigate to the Azure ML workspace named **mlw-dp100-labs**.
2. Select the Azure ML workspace, and in its Overview page, select Launch studio. Another tab will open in your browser to open the Azure ML studio.
3. Close any pop-ups that appear in the studio.
4. Within the Azure ML studio, navigate to the **Compute** page and verify that the compute instance and cluster you created in the previous section exist. The compute instance should be running, the cluster should be in Succeeded state and have 0 nodes running.

###### Use the Python SDK to train a model

Now that you have verified that the necessary compute has been created, you can use the Python SDK to run a training script. You will install and use the Python SDK on the compute instance and train the ML model on the compute cluster.

1. In your **compute instance**, there are a number of options in the **Applications** field. Select the **Terminal** application to launch the terminal 

2. In the terminal, install the Python SDK on the compute instance by running the following commands in the terminal:

   ```
   pip uninstall azure-ai-ml
   pip install azure-ai-ml
   ```

3. Run the following command to clone a Git repository containing notebooks, data, and other files to your workspace:

   ```
    git clone https://github.com/MicrosoftLearning/mslearn-azure-ml.git azure-ml-labs
   ```

4. When the command has completed, in the **Files** pane, select **↻** to refresh the view and verify that a new **Users/\*your-user-name\*/azure-ml-labs** folder has been created.

5. Open the **Labs/02/Run training script.ipynb** notebook. (Select **Authenticate** and follow the necessary steps if a notification appears asking you to authenticate.)

6. Verify that the notebook uses the **Python 3.10 - AzureML** kernel on the upper right corner of the notebook environment. Each kernel has its own image with its own set of packages pre-installed.

7. Run all cells in the notebook.

A new job will be created in the Azure Machine Learning workspace. The job tracks the inputs defined in the job configuration, the code used, and the outputs like metrics to evaluate the model.

###### Review your job history in the Azure ML studio

When you submit a job to the Azure Machine Learning workspace, you can review its status in the Azure Machine Learning studio.

1. Either select the job URL provided as output in the notebook, or navigate to the **Jobs** page in the Azure Machine Learning studio.

2. A new experiment is listed named **diabetes-training**. Select the latest job **diabetes-pythonv2-train**.

3. Review the job’s **Properties**. Note the job **Status**:

   - **Queued**: The job is waiting for compute to become available.
   - **Preparing**: The compute cluster is resizing or the environment is being installed on the compute target.
   - **Running**: The training script is being executed.
   - **Finalizing**: The training script ran and the job is being updated with all final information.
   - **Completed**: The job successfully completed and is terminated.
   - **Failed**: The job failed and is terminated.

   4. Under **Outputs + logs**, you’ll find the output of the script in **user_logs/std_log.txt**. Outputs from **print** statements in the script will show here. If there’s an error because of a problem with your script, you’ll find the error message here too.
   5. Under **Code**, you’ll find the folder you specified in the job configuration. This folder includes the training script and dataset.

   ###### Delete Azure resources

   When you finish exploring Azure Machine Learning, you should delete the resources you’ve created to avoid unnecessary Azure costs. 

   1. Close the Azure Machine Learning studio tab and return to the Azure portal.
   2. In the Azure portal, on the **Home** page, select **Resource groups**.
   3. Select the **rg-dp100-labs** resource group.
   4. At the top of the **Overview** page for your resource group, select **Delete resource group**.
   5. Enter the resource group name to confirm you want to delete it, and select **Delete**.

### Training: Explore and configure the Azure Machine Learning workspace, Module: Make data available in Azure Machine Learning













