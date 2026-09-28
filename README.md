# STADIO CAP182

## SS1

# Part A - Motivation
StadioEquities is a retail investment platform which allows consumers to invest into different stocks and equities with a lower barrier of entry than traditional stockbrokers. The business is suffering from multiple pain points. 

One of these pain points is the current client experience. Clients are making increasingly higher amounts of queries as their accounts are frozen due to suspicious or possible fraudulent behavior. This is a frustrating experience for clients and internal client management teams. Once accounts are frozen our internal teams need to review and investigate the reason behind the suspicious flag. This costs the business time and resources, especially when many of the transactions being flagged are not fraudulent. The high volume of queries is causing the internal teams capacity to shrink as much of their attention is spent on resolving false positive fraudulent account queries. This is slowing down their overall turnaround times. Clients are unhappy that their accounts are frozen as the hold suspension on their accounts stops them from transacting on the platform. 

The business risks losing clients due to their frustrations of poor query resolution turnaround times and frequent account suspensions stopping them from investing freely. 

The business has the goal of “Protecting clients from themselves and from harm“ and this project will help achieve that. The business will be able to solve the issue of the high volume transactions being flagged as fraudulent. The client-service team will be able to focus on and effectively investigate highly suspicious transactions rather than spending time investigating high volumes of safe transactions. Reducing the amount of false positive transactions flagged will mean that less queries will come through to the client-service department. This means that they will be able to handle their capacity effectively and be prepared for if the client base was to increase.

Our goal is to create a data model which accurately flags transactions as fraudulent, reducing false positives and reducing the load put on the client-service department. 
# Part B - Problem Statement
StadioEquities platform is a retail investment platform operating in South Africa aiming to make investing in equities more accessible for the general public. It is a digital investment platform which allows investors to begin investing with a low barrier to entry. StadioEquities generates revenue from clients actively trading, buying and selling on their platform.

The client-service department is experiencing a growing volume of client queries. Many of which are related to client accounts being adversely impacted by false fraud alerts. This is caused by the current system flagging transactions as fraudulent. Currently the system has a high number of false positives. This leads to client accounts being frozen for suspicious activity. These delays cause stress among clients who are genuinely attempting to utilize the platform.

The frustrating experience is causing existing clients to open up queries at high volumes, overwhelming the client-service team and causing delays in all client query processing times. Clients may churn due to their bad experience with the platform leading to the business losing possible revenue.

We are aiming to create a predictive model which would be able to effectively flag transactions as fraudulent. The model should aim to reduce false positives and increase true positives so that we do not lose actual fraud in the noise. 

# Part E - RAAIDD Log
| RAAID | Description |
| :--- | :--- |
| Risks  | We are working with customer and transactional data. We are risking compliance. Since client information is sensitive we could possibly have a leak of client information or a breach of the POPI act. We may be working with demographic data and we risk our model becoming biased to a specific region or client profile.|
| Actions  | We will need to prepare and analyze our data accordingly, generate a binary classification model to determine the probability of a fraudulent transaction and evaluate the performance of our model. |
| Assumptions | I believe that since we are working with fraud data that there will be a great class imbalance as fraud is a rare occurrence. I assume that a model such as a neural network may work well for this type of problem since it can handle many categorical variables well. |
| Issues  | High numbers of false positives for the current model used to flag transactions as fraudulent. |
| Decisions  | During the generation and evaluation, we will be prioritizing reducing the count of false negatives. This will determine our evaluation and model recommendation.  |
| Dependencies  | I need to create new data features within my dataset. I will be using aggregated values such as average transaction amount. This will be highly dependent on what data I receive from the client. |

[View Data Request](Datasets/Data_Request.pdf)


## SS2
