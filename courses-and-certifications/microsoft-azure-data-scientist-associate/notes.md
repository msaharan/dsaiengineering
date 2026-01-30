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

### Explore and configure the Azure Machine Learning workspace

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

