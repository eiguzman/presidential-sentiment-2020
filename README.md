# presidential-sentiment-2020
A sentiment analysis project on presidential candidate speeches during the 2020 election.

## Abstract

This study examines the impact of the COVID-19 pandemic on the rhetoric of U.S. presidential candidates Donald Trump and Joe Biden during the 2020 election cycle, utilizing a corpus of over 700 publicly broadcasted campaign speeches from January 2019 to January 2021. Moving beyond traditional social media analyses, this research leverages a novel dataset of broadcast speeches to explore shifts in thematic focus and sentiment related to COVID-19. The analysis comprises three foundational components: a net sentiment analysis, Latent Dirichlet Allocation (LDA) for topic modeling, and a prevalence analysis of COVID-19-related content. A qualitative categorization of speech snippets referencing COVID-19 was developed to further highlight the prevalence of the topic and was used to train a logistic regression classifier. The model achieved approximately 74% accuracy when predicting on COVID-related snippets, with results indicating statistically significant change in sentiment before versus after the pandemic became central to public discourse. These findings suggest that, within televised speeches, presidential rhetoric shifted to be more relatable to the issues the public faced concerning COVID-19, contrasting with the more negative sentiment observed in social media platforms. Overall, this research demonstrates the utility of broadcast speech analysis and text modeling techniques in understanding political communication dynamics during unprecedented crises.

## Introduction

The COVID-19 pandemic has profoundly reshaped societal perspectives on how to engage in public discourse during times of health outbreaks. Stay-at-home orders, masking, and social distancing required politicians to rethink their campaign strategies across various platforms. While numerous studies have analyzed social media content to gauge shifts in political sentiment and messaging, less attention has been given to formal, broadcasted speeches delivered by presidential candidates, which often serve as a primary source of official communication and policy positioning when public forums are unavailable. This project investigates how the rhetoric of Donald Trump and Joe Biden evolved during the 2020 U.S. presidential election, with a particular focus on the impact of COVID-19. By analyzing over 700 publicly available campaign speeches spanning January 2019 to January 2021, this porject aims to determine whether and how candidates' messaging regarding the pandemic changed as it transitioned from a distant health crisis to a central political, social, and economic issue. This study seeks to uncover shifts in thematic focus and emotional tone to understand political communication dynamics during crises and highlights differences in rhetoric across political candidates.

For this project, I have used a text dataset of campaign speeches of the main tickets in the 2020 US presidential election . These speeches are from public speeches delivered by Republican candidate Donald Trump and Democratic candidate Joe Biden via public broadcasting sources such as C-SPAN. Over 700 speeches were recorded, some of which are duplicate broadcasts by different channels. While some speeches are short briefings, many speeches span for minutes, causing large size discrepancies between speeches. To normalize length, speeches are broken down into chunks of 250 words. The implications of this change are that entire speeches will no longer be classified as being related to COVID if only a subsection mentions keywords related to the topic. Our sentiment analysis will show graphs based on full speeches as a reference to the graph based on the subsections. The time period covers speeches between January 2019 through January 2021. Many of our tests will compare text by each president before and after February 11, 2020, the day the World Health Organization (WHO) declared the official disease name "COVID-19".

Background reading reveals that many studies have been performed in 2021 solely on Twitter data on presidential posts. This data is numerous and scraping using the Twitter API was simpler. Nowadays, X prevents standard users from using its API to scrape posts in an effort to combat bot activity as well as to protect a potential monetizable data source. The data source I am using is relatively new, having been created one year ago. Xia et al. analyzes over 260,000 Twitter posts related to the 2020 U.S. presidential election using multi-layer perceptrons, finding that negative sentiment was more prevalent than positive, and demonstrates that social media sentiment trends can reveal key political events and opinions (2021). Ali et al. analyze 7.6 million tweets related to the 2020 US presidential election to reveal how deleted, suspended, and accessible tweets differ in public sentiment towards candidates, emphasizing the importance of including inaccessible posts for accurate opinion assessment (2022).

## Question

What impact did COVID-19 have on the rhetoric of presidential candidates? Do we see a change in speeches before the event became mainstream news? COVID-19 and Statewide Shutdowns led to a major shift in the political climate. Sixty-two percent of sampled voters said the coronavirus outbreak was a top issue in the 2020 election. We will attempt to determine if presidential candidates shifted their campaign platform based on this issue.

## Hypothesis

To analyze the impact of COVID-19 on the rhetoric of presidential candidates, I will develop a categorization based on the thematic focus of their speeches concerning COVID-19. Each speech snippet that mentions related topics such as COVID-19, the pandemic, or vaccines will be coded into this category. This approach allows for a nuanced understanding of how candidates’ messaging shifted before and after COVID-19 became a prominent issue in public broadcasting, providing insight into whether their rhetoric became more empathetic, accusatory, optimistic, or pessimistic over time. As such, our hypothesis is written below:

> $H_0$: The probability of a speech being related to COVID-19 does not depend on whether the speech was delivered before or after February 11, 2020.
> 
> $H_1$: The probability of a speech being related to COVID-19 increases if it was delivered after February 11, 2020.

To test this hypothesis, we will implement Pearson's Chi-squared test with Yates' continuity correction since we are working with qualitative data.

## Exploratory Data Analysis

Before this analysis can be made, we must first perform three prior analyses: A net sentiment analysis of speeches by president, a Latent Dirichlet allocation and sentiment analysis grouped by speech topic, and a topic prevalence analysis of COVID-19 in speeches. These analyses will corroborate our findings that show there are some shifts in presidential speech platforms and will be the foundation of our logistic regression model weights in this project and any future improvements. This categorization will serve as a qualitative measure to capture the shifts in rhetoric, enabling comparison across different time periods and candidates. By examining the prevalence of each tone before and during COVID-19, I can assess whether candidates adapted their messaging to reflect the evolving political and social climate. As this was an individual project, inter-coder reliability and conflict was a non-issue, as all code was streamlined and validated to work before moving onto the next analysis.

### Sentiment Analysis

Sentiment analysis is a technique used in natural language processing (NLP) to determine the emotional tone or attitude expressed within textual data. It involves computationally identifying and classifying opinions, sentiments, or emotions conveyed in wordsin a documents, typically as positive, negative, or neutral. In political discourse, sentiment analysis can reveal shifts in tone, public opinion, or strategic messaging. By quantifying these emotional cues, researchers can assess how politicians' rhetoric evolves over time, especially in response to major events, providing insights into their communication strategies and public engagement.

The goal of this section is to analyze the sentiment scores of presidential candidates Joe Biden and Donald Trump over their speech history. This is a primary indicator of overall sentiment over time before and after the mainstream discussion of COVID-19. By analysing the regression lines of each candidate, significant change in sentiment can be discovered by comparing the shift over the y-axis at our chosen time point. The script loads and preprocesses speech data, splits lengthy speeches into manageable chunks, and constructs text corpora for each candidate. It then performs sentiment analysis by calculating a net sentiment score based on positive and negative word counts within each speech segment. The resulting sentiment scores are visualized over time with plots highlighting the period before and after the COVID-19 outbreak became prominent, enabling an examination of potential shifts in tone or emphasis related to COVID-19 in their public communications.

![Joe Biden's speech sentiment values over time](img/biden_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 1: Joe Biden's speech sentiment values over time</i></small>
</div>
<br/>

![Donald Trump's speech sentiment values over time](img/trump_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 2: Donald Trump's speech sentiment values over time</i></small>
</div>
<br/>

![Combined sentiment values over time](img/combined_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 3: Combined sentiment values over time</i></small>
</div>
<br/>

Each dot in the scatterplot represents a segment of a speech. The x-axis represents the date in which the speech was televised, and the y-axis represents the sentiment score of that speech fragment. Two regression lines are calculated with their split along February 11, 2020. Additionally, sentiment scores are color coded to highlight which particular sections were more extreme in positivity or negativity. Both presidential candidate's graphs are then superimposed to normalize the scale of the y-axis. Our original project performed sentiment analysis on the entire corpus as a whole. Results are shown below to provide an alternate view of total sentiment across a speech.

![Joe Biden's sentiment values of full corpora over time](img/images/biden_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 4: Joe Biden's sentiment values of full corpora over time</i></small>
</div>
<br/>

![Donald Trump's sentiment values of full corpora over time](img/images/trump_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 5: Donald Trump's sentiment values of full corpora over time</i></small>
</div>
<br/>

![Combined sentiment values of full corpora over time](img/images/combined_sentiment_over_time.png)
<div style="text-align: center;">
<small><i>Figure 6: Combined sentiment values of full corpora over time</i></small>
</div>
<br/>

The graphs above allow us to infer a few interesting observations about the text:
When speeches are split into sections, the regression lines lose their significance, as many sections are neutral in sentiment. Calculating sentiment of whole documents provides a better understanding of the general tone of the speech. Regression lines see a significant jump down after February 11, 2020, signaling that topics were generally less positive in response to the societal changes brought about by the pandemic. A positive slope for the second regression line for both candidates signifies that sentiment gradually increased as breakthroughs were being made to combat the virus.

Additionally, we can identify from Figure 6 that Donald Trump's highest sentiment scores are overwhelmingly higher than Joe Biden's most positive speech. Why would this be? In order to answer this question, we must perform LDA to identify the primary topic of each speech.

### Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is a statistical modeling technique used in natural language processing to discover underlying topics within large collections of text data. Speeches consist of general campaigning strategies to more specific topics such as healthcare, global policy, the economy, immigration, labor, and the COVID-19 outbreak. It operates on the assumption that each document is a mixture of multiple topics, and each topic is characterized by a distribution of words. By analyzing the co-occurrence patterns of words across documents, LDA infers these latent topics, allowing researchers to interpret the themes present in the text corpus. The model outputs probability distributions for each document over topics and for each topic over words, enabling a nuanced understanding of the thematic structure within the data.

The goal of this section is to analyze changes in political discussion by each candidate through their speeches. We appliy LDA to identify 30 underlying topics in their speeches over time. The reason 30 topics was chosen was because it landed in a sweet spot between topic broadness, topic speficity, and code complexity. A lower number resulted in topics that were too broad, such as all healthcare topics being recognized as COVID-related. Likewise, a higher number resulted in topics that were too specific to be significant, such as Campaigning in Illinois versus Campaigning in Pennsylvania, and resulted in too much clutter when graphed. Our project then assigns each speech segment to the most relevant topic and visualizes how the candidates' focus shifted, especially toward COVID-related topics, by examining sentiment trends within these topics before and after the pandemic's rise. This analysis aims to determine whether and how the candidates' campaign messaging shifted to emphasize COVID-19 in response to its increasing importance in public and political discourse.

![Joe Biden's Speeches categorized by topic](img/biden_sentiment_clustered.png)
<div style="text-align: center;">
<small><i>Figure 7: Joe Biden's speeches categorized by topic</i></small>
</div>
<br/>

![Joe Biden's Speeches related to COVID have been highlighted](img/biden_covid_topics.png)
<div style="text-align: center;">
<small><i>Figure 8: Joe Biden's speeches related to COVID have been highlighted</i></small>
</div>
<br/>

![Donald Trump's Speeches categorized by topic](img/trump_sentiment_clustered.png)
<div style="text-align: center;">
<small><i>Figure 9: Donald Trump's speeches categorized by topic</i></small>
</div>
<br/>

![Donald Trump's Speeches related to COVID have been highlighted](img/trump_covid_topics.png)
<div style="text-align: center;">
<small><i>Figure 10: Donald Trump's speeches related to COVID have been highlighted</i></small>
</div>
<br/>

Topics directly related to COVID-19 are highlighted in their own graphs, showcasing how often the topic was brought up by each candidate. We notice a few interesting observations in these graphs. First, there are a few speeches in Joe Biden's cluster graph that are identified as being COVID-related, despite Biden's first official statement on COVID-19 being broadcast on March 12, 2020. These are misclassifications by our LDA, incorrectly classified as being COVID-related due to using words such as epidemic (most of these misclassifications were actually discussing the *guns and mass shootings* epidemic). These mistakes, although easily fixable by hand, are left in to show that these models are not perfect, and that even with these misclassifications, they are not significantly damaging to the accuracy of our results. Below we also display the results from the original project, where clusters were generated based on the entire speech corpus.

![Original topic clusters of Biden's speeches](img/images/biden_sentiment_clustered.png)
<div style="text-align: center;">
<small><i>Figure 11: Original topic clusters of Biden's speeches</i></small>
</div>
<br/>

![Original topic clusters of Trump's speeches](img/images/trump_sentiment_clustered.png)
<div style="text-align: center;">
<small><i>Figure 12: Original topic clusters of Trump's speeches</i></small>
</div>
<br/>

Now we can identify why Donald Trump's speeches are overwhelmingly positive. When speeches are clustered as a whole, the primary topic of many of Trump's speeches involve the public opinion. Looking closer at the text documents reveal that in most speeches, Trump will speak positively about himself and of his accomplishments the past four years as president, as if it was the opinion of the average voter. Additionally, Trump's campaign slogan "Make America **Great** Again" aids in improving the sentiment score each time the phrase is mentioned.

Alternatively, Joe Biden's speeches cluster closer to the intercept due to "Negative Campaigning". Negative campaigning in presidential elections refers to the strategy of attacking or criticizing an opponent rather than promoting one's own policies and strengths. This approach aims to undermine an opponent's character, record, or credibility to sway voters by emphasizing their flaws or perceived weaknesses. While it can energize a candidate's base and draw attention to issues, negative campaigning often sparks controversy, potentially fostering voter cynicism or apathy. Despite criticisms of its divisiveness, many campaigns use negative tactics to diminish an opponent’s electability, making it a common feature of presidential elections.

### Topic Prevalence

Topic prevalence analysis involves quantifying how dominant specific topics are within a collection of documents over time or across different groups. This process typically begins with applying a topic modeling technique such as LDA to identify underlying thematic structures in the text data. Once a model is fit, each document receives a distribution of topics, indicating the proportion of content related to each theme. These proportions can be summarized, visualized, and compared across time or categories to understand trends, shifts, or the prominence of particular issues. In our case, we will focus on the prevalence of COVID-19 references in speeches over the course of the election cycle. By examining the proportion of specific topics within speeches, researchers can infer how candidates prioritize or shift focus on different themes throughout their campaigns or responses to events.

The goal of this section is to understand how prevalent COVID-19 speeches were for each candidate after the observation date. First, after fitting LDA models for each candidate, the code assigns the most probable topic to each speech document and associates descriptive topic labels. It then constructs matrices of posterior topic probabilities for each speech, enabling the calculation of combined proportions of related COVID-19 topics by summing relevant topic columns. For example, 4 topics in Biden's speeches were COVID-related, while Trump's speeches only generated 1 topic. These proportions are added to the speech data, which is then merged across candidates for comparative visualization. The plotting segment uses `ggplot2` to create scatter plots and smoothed trend lines over time. This approach facilitates a clear understanding of thematic focus shifts within the speeches over the campaign timeline.

![Topic Prevalence of COVID over time](img/covid_prevalence_over_time.png)
<div style="text-align: center;">
<small><i>Figure 13: Topic Prevalence of COVID over time</i></small>
</div>
<br/>

Once again we notice a few errors in our topic assignment methods. We can see in Figure 13 that there is one outlier before the observation date that is associated with one of Biden's speech segments. Drawing parallels from our previous explanation, we can safely consider this outlier not significant to changing the results of our hypothesis test. Additionally, we notice our most significant trend in this graph. We can infer from this graph that Joe Biden's speeches -initially full of statements of his leadership qualities and capabilities- along with his campaign strategy shifted towards bringing up COVID-19, the pandemic, stay-at-home orders, vaccine research, and the sitting president's struggle to contain the virus effectively. Donald Trump's strategy was not effected as much as expected, with his campaign strategy focused on public rallies to energize the support of his followers while highlighting the economic strengths of the United States pre-pandemic.

## Methods

### Hand Coded Classification

In order to guarantee classification labels for each speech, we need to manually read each document to determine if there was any mention of COVID-related topics discussed. Originally, our project did not fragment speeches to smaller subsections. This made it difficult to properly classify the relevancy of COVID in these speeches, as text samples with over 2000 words would be classified as being COVID-related if even one sentence made a reference. With text broken down to manageable sizes, we can now determine if specific sections of a speech discussed the pandemic more appropriately. 

Our hand coded data consists of 100 documents, 50 of them from each presidential candidate to maintain sample consistency. To simplify the process, a python script is run to determine if a text corpus contains sertain keywords and classifies the text based on its findings. This was verified to be accurate by hand coding a sample of similar size and getting identical results. Using our hand coded data, we developed a model using K-Fold Cross validation, training and validating our known documents 5 times. We fit our data and predicted using logistic regression. This form of classification is simple to perform on small datasets, and we are able to use model weights that have been developed in earlier sections of our project, such as net sentiment, topic name, and prevalence value of covid topics.

Each validation set averaged 75% accuracy on the positive classification. Accuracy for this classification was compared against hand-coded documents that were assigned a 1 if there was at least one occurrence of the words “covid-19”, “coronavirus”, “pandemic”, or “vaccine”. The word “shutdown” and related words were not counted, as shutdowns were stagnant, with New York being the first state to issue shutdown orders. Additionally, the word shutdown could be in reference to the shutdown of an industry due to reasons not related to the pandemic.

Our test data was fit onto the highest scoring validation set, which scored a .749 accuracy. Results of our classification are shown in the image above. Our test data was compared in the same manner as the validation sets, where a python script classified each speech based on keywords, even if the topic did not encompass the entire corpus. Should we choose to use only the speeches whose majority topic was about COVID, we would have an insufficient number of documents to produce significant results. Our confusion matrix shows a significant quantity of false positives and false negatives; however, for a classifier that had to choose between 20+ topics, an accuracy greater than or equal to 75% signifies that the model is working properly. Our recall is also higher than our precision, as false negatives are more important to classify correctly. Despite having a significant p-value, a confusion matrix does not accurately reflect any results. As such, we will move onto a bette rhypothesis test below.

![Confusion Matrix of hand coded classes](img/cm.png)
<div style="text-align: center;">
<small><i>Figure 14: Confusion Matrix of hand coded classes</i></small>
</div>
<br/>

### Chi-squared Test

The Chi-squared test with Yates' continuity correction is a statistical method used to assess whether there is a significant association between two categorical variables. In this case, we measure the timing of speeches before or after February 11, 2020 and the presence or absence of COVID-related topics. The correction adjusts the traditional Chi-squared test to account for the discrete nature of the data, especially when sample sizes are small, reducing the risk of overestimating statistical significance. This correction modifies the calculation of the difference between observed and expected frequencies, making the test more conservative and less prone to Type I errors. When applied, the test evaluates whether any observed differences in the proportion of COVID-related topics before and after the specified date are statistically meaningful or could have occurred by chance.

The goal of this section is to compare the frequency of COVID-related topics in speeches by Biden and Trump before and after February 11, 2020, to determine if there was a significant change in the emphasis on COVID-19 in their rhetoric over time. By constructing contingency tables for each candidate and performing the Chi-squared test with Yates' correction, the analysis aims to identify whether the proportion of COVID-related content significantly increased or decreased, providing insights into how each candidate's messaging evolved during the early stages of the pandemic. This statistical approach helps to quantify shifts in discourse and assess whether these changes are statistically significant rather than due to random variation.

![Results of Chi2 test on Biden's speeches](img/chi2_biden.png)
<div style="text-align: center;">
<small><i>Figure 15: Results of Chi2 test on Biden's speeches</i></small>
</div>
<br/>

![Results of Chi2 test on Trump's speeches](img/chi2_trump.png)
<div style="text-align: center;">
<small><i>Figure 16: Results of Chi2 test on Trump's speeches</i></small>
</div>
<br/>

## Results

The results indicate highly significant associations between speech timing and COVID-19 topic relevance for both candidates (p-values < 2.2e-16). Specifically, for Biden, COVID-related topics surged to 345 out of 1632 speeches, suggesting a marked increase in COVID-19 coverage during the pandemic period. Similarly, Trump’s COVID-related mentions increased to 196 out of 3280 speeches after February 11. We reject the null hypothesis stated at the beginning of our project. These findings support the alternate hypothesis that the probability of COVID-19 being a topic in presidential speeches increased following the pandemic onset date, reflecting a substantial shift in discourse focus.

The results of our model change if we decide to use all topic names as one-hot encoded values. We were unable to implement this in our project, as too many errors regarding the logistic regression equation occurred. Instead, topic names were omitted from the analysis. Additionally, the rhetoric of presidential speeches that are broadcast to the public are generally more positive, as these are widely viewed by most audiences through news sources or through broadcasts. If we were to implement sentiment analysis on other forms of media, such as tweets, we would see significantly more negative rhetoric on the subject of the pandemic.

## Conclusion

In conclusion, this project developed multiple analyses from our data, from identifying net sentiment analyses of each president’s speech patterns to identifying whether presidential rhetoric changed during the COVID-19 pandemic. This project has taught us useful skills related to implementing text-as-data projects using R. in comparison to other programming languages, R provides an efficient programming setting with packages that optimize performance for modifying and analyzing data structures. By processing data using Latent Dirichlet allocation and structural topic models, summarization of text corpora can help identify hidden patterns that are not found via conventional text reading.

## Works Cited

Chalkiadakis, I., Anglès d'Auriac, L., Peters, G., & Frau-Meigs, D. (2025). A text dataset of campaign speeches of the main tickets in the 2020 US presidential election [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.14785782.

Ethan Xia, Han Yue, and Hongfu Liu. 2021. Tweet Sentiment Analysis of the 2020 U.S.
Presidential Election. In Companion Proceedings of the Web Conference 2021 (WWW '21). Association for Computing Machinery, New York, NY, USA, 367–371. https://doi.org/10.1145/3442442.3452322

Ali, R.H., Pinto, G., Lawrie, E. et al. A large-scale sentiment analysis of tweets pertaining to
the 2020 US presidential election. J Big Data 9, 79 (2022). https://doi.org/10.1186/s40537-022-00633-z

<br><br><br>
*Originally a final project for DSC 161 at the University of California, San Diego*