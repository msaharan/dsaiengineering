[Mohit Saharan](https://linkedin.com/in/msaharan), P9, 20260423

___

# Understanding tabular foundation models: trying TabPFN Client (cloud solution)

Today I ran the same notebook as [yesterday](https://www.linkedin.com/posts/msaharan_20260422-understanding-tfms-tabpfn-handson-demopdf-ugcPost-7452807833286352897-46Wg?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g) but differently. Yesterday I ran it in Google Colab and TabPFN-related calculations were performed locally using Colab's GPU. Today I first tried to run it on my MacBook using TabPFN Client, which allows offloading TabPFN-related computations to its cloud by sending data there, but I failed at installation stage because Conda refused to install some dependencies. Then I went back to Google Colab. The following content is for the experiments conducted within Colab.

Even though Colab had GPU available, I again tried to use TabPFN Client there to save my free GPU allowance in Colab. The user experience surprised me. I concluded that the only way to use TabPFN for now is with a local GPU. In this post, I discuss why think so.

## TabPFN Client

### API Usage limits

![Screenshot 2026-04-23 at 20.12.00](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2020.12.00.png)

The Client usage is limited by daily API usage allocation. At the time of taking this screenshot, I was running the notebook from yesterday, and I was at the end of it, running SHAP calculations.  Before this step, the notebook had cells where the performance of four models (including TabPFN) was compared using a 200-row dataset for classification and a 500 row dataset for regression. In both cases, a five times repeated five-fold cross validation was used. For classification, the computations consumed almost 1.5 million tokens. I didn't track the consumption in the case of regression. However, seeing 9 million tokens consumed by a toy analysis, compared to day-to-day work of a data scientist, indicated to me that real data science work would consume hundreds of millions of tokens every day. I think the current API usage limit is only a rough threshold that the company decided to set for now, and there's also an option to request an increased limit on the website. However, in the next section, I argue that it might not be needed after all, for now.

### Slow cloud computing

I found that offloading calculations to cloud using TabPFN Client is very slow compared to doing them locally. The following picture shows the output of the cell in which the performance of four models is compared for the binary classification task. Here, the T4 runtime type was used in combination with TabPFN Client. The cell took almost 3 minutes to finish calculations and an additional minute to generate the summary table and the plot.

![Screenshot 2026-04-23 at 20.57.43](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2020.57.43.png)

The following picture shows the result of the same cell when I used the T4 runtime type and the local GPU for TabPFN. This time, the calculations completed in only 33 seconds, and an additional minute was required again to produce the summary table and the plot.

![Screenshot 2026-04-23 at 21.11.55](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2021.11.55.png)

I saw the most drastic difference in the last cell, where SHAP-related calculations are performed to interpret the model's performance. The following picture was taken when I initially ran the notebook using the CPU runtime type in combination with TabPFN Client. To be consistent with the comparison of the binary classification cell shown above, I also ran the SHAP cell again using the T4 runtime type, and didn't see a noticable difference within 5 minutes of running, so the performance shown in the following picture should also hold for the T4 runtime type. The observation of interest here is that the cell was almost half-way through even after running for 25 minutes.  

![Screenshot 2026-04-23 at 20.23.56](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2020.23.56.png)

![Screenshot 2026-04-23 at 20.25.59](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2020.25.59.png)

The following picture was taken when I re-ran the cell using the T4 runtime type and the local GPU for TabPFN. This time, it the cell produced all results within 2 minutes. 

![Screenshot 2026-04-23 at 21.16.18](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2021.16.18.png)



### Perhaps cloud computing could be made faster

The results shown above indicate that TabPFN Client is perhaps not ready for use, as of now, due to its slow speed. I don't know why it is slow and whether it can be sped up, but I do have an observation to share. The following picture shows the history of requests made by my notebook to the API.

![Screenshot 2026-04-23 at 21.59.11](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2021.59.11.png)

It seems to be making requests for a few hundred milliseconds every few seconds. I don't know how to interpret these timings at more depth, but perhaps the experts can and the data transmission framework could be made more intelligent here.

## Migrated to Kaggle

Colab provides very little free usage quota. Mine is running out already, so I migrated the notebook to Kaggle. The notebook ran fine, as shown in the following pictures, and took almost the same time as Colab. The overall user experience on Kaggle is much better than the free Colab account. Free Colab account feels very laggy. Kaggle feels smooth. 

![Screenshot 2026-04-23 at 22.25.00](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2022.25.00.png)

![Screenshot 2026-04-23 at 22.28.44](./20260423-understanding-tabular-foundation-models-trying-tabpfn-client.assets/Screenshot%202026-04-23%20at%2022.28.44.png)

## Outlook

Today I made my work environment reliable. I will continue from here tomorrow. 