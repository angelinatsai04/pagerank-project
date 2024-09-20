<!--
**Computation:**
This project has low computational requirements.
You should be able to complete it on your own laptops.
-->

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.
>
> In python, we can use the built-in `gzip` module to access gzipped files.
> The following python code is equivalent to the bash code above:
>
> ```
> >>> import gzip
> >>> fin = gzip.open('data/small.csv.gz', mode='rt')
> >>> print(fin.read())
> source,target
> 1,2
> 1,3
> 3,1
> 3,2
> 3,5
> 4,5
> 4,6
> 5,6
> 5,4
> 6,4
> ```
>
> There are many terminal commands throughout these instructions.
> If you haven't used the terminal before, and so these commands are unfamiliar, that's okay.
> I'd be happy to explain them in office hours,
> or there are many tutors in the QCL available who can help.
> (There are no tutors for this class specifically, but anyone who has taken CSCI046 or CSCI133 with me will be able to help with the terminal.)
>
> Furthermore, you don't "need" to understand the terminal commands in detail,
> since you are not required to run these commands or to create your own.
> The important part is to understand the English language description of what the commands are doing,
> and to understand that this is just how I computed what the English language text is describing.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the $P$ matrix,
this is also the value of $\text{nnz}(P)$.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of $P$, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the [`FIXME: Task 1` annotation](https://github.com/mikeizbicki/cmc-csci145-math166/blob/81ed5d2b75f5bc23b8de93805c29321ab431ed9b/topic01_computation_pagerank/project/pagerank.py#L144).

> **NOTE:**
> The power method is the only data mining algorithm you will implement in class.
> You are implementing it because there are no standard library implementations available.
> Why?
> 1. The runtime is heavily dependent on the data structures used to store the graph data.
>    Different applications will need to use different data structures.
> 1. It is "trivial" to implement.
>    My solution to this homework is <10 lines of code.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to minor implementation details and weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

> **NOTE:**
> There are no automated test cases to pass for this assignment.
> Test cases for algorithms involving floating point computations are hard to write and understand.
> Minor-seeming implementations details can have large impacts on the final result.
> These software engineering issues are beyond the scope of this class.
>
> Instructions for how I will grade your homework are contained in the [submission section](#submission) at the end of this document.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their $P$ matrix for the web,
they use a similar (but much more complicated) process to modify the $P$ matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<https://www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.

> **NOTE:**
> It should be "obvious" to you that large alpha values imply that the structure of the webgraph has more influence on the final result,
> and small alpha values ignore the structure of the webgraph.
> Recall that the word "obvious" means that it follows directly from the definition,
> but you may still need to sit and meditate on the definition for a long period of time.

If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).
    Ensure that all of the project files are copied from this folder into your new repo.

1. As you complete the tasks above:
    Run the corresponding commands below, and paste their output into the code blocks.
    Please ensure correct markdown formatting.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
    DEBUG:root:i=0 residual=0.2562914788722992
    DEBUG:root:i=1 residual=0.11841227114200592
    DEBUG:root:i=2 residual=0.07070132344961166
    DEBUG:root:i=3 residual=0.03181540593504906
    DEBUG:root:i=4 residual=0.020496580749750137
    DEBUG:root:i=5 residual=0.010108358226716518
    DEBUG:root:i=6 residual=0.006371477618813515
    DEBUG:root:i=7 residual=0.003422832814976573
    DEBUG:root:i=8 residual=0.0020879278890788555
    DEBUG:root:i=9 residual=0.0011749786790460348
    DEBUG:root:i=10 residual=0.0007013090653344989
    DEBUG:root:i=11 residual=0.00040324186556972563
    DEBUG:root:i=12 residual=0.00023794587468728423
    DEBUG:root:i=13 residual=0.00013811791723128408
    DEBUG:root:i=14 residual=8.111781789921224e-05
    DEBUG:root:i=15 residual=4.723723031929694e-05
    DEBUG:root:i=16 residual=2.7683758162311278e-05
    DEBUG:root:i=17 residual=1.6175998098333366e-05
    DEBUG:root:i=18 residual=9.42118731472874e-06
    DEBUG:root:i=19 residual=5.5282102948694956e-06
    DEBUG:root:i=20 residual=3.2534051115362672e-06
    DEBUG:root:i=21 residual=1.8578108438305208e-06
    DEBUG:root:i=22 residual=1.1547198255357216e-06
    DEBUG:root:i=23 residual=6.183532832437777e-07
    INFO:root:rank=0 pagerank=6.6270e-01 url=4
    INFO:root:rank=1 pagerank=5.2179e-01 url=6
    INFO:root:rank=2 pagerank=4.1434e-01 url=5
    INFO:root:rank=3 pagerank=2.3175e-01 url=2
    INFO:root:rank=4 pagerank=1.8590e-01 url=3
    INFO:root:rank=5 pagerank=1.6917e-01 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
    INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 pagerank=8.9228e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 pagerank=7.0394e-04 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=3 pagerank=6.9157e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=4 pagerank=6.7045e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=5 pagerank=6.6260e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=6 pagerank=6.5050e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=7 pagerank=6.3623e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
    INFO:root:rank=8 pagerank=6.1252e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
    INFO:root:rank=9 pagerank=6.0191e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
    INFO:root:rank=0 pagerank=5.7827e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=5.2340e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=2 pagerank=5.1298e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=4.6601e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=4 pagerank=4.5935e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=5 pagerank=4.3073e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=4.0936e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=3.7592e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=3.4510e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=3.4486e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
    INFO:root:rank=0 pagerank=4.5747e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
    INFO:root:rank=1 pagerank=4.4175e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 pagerank=2.6929e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
    INFO:root:rank=3 pagerank=1.9392e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
    INFO:root:rank=4 pagerank=1.5453e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
    INFO:root:rank=5 pagerank=1.5358e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
    INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
    INFO:root:rank=7 pagerank=1.4222e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
    INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
    INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
    INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
    INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
    INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
    INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
    DEBUG:root:i=0 residual=1.3793749809265137
    DEBUG:root:i=1 residual=0.11642682552337646
    DEBUG:root:i=2 residual=0.07496178895235062
    DEBUG:root:i=3 residual=0.03170211240649223
    DEBUG:root:i=4 residual=0.017446598038077354
    DEBUG:root:i=5 residual=0.00852623675018549
    DEBUG:root:i=6 residual=0.004441831726580858
    DEBUG:root:i=7 residual=0.0022433067206293344
    DEBUG:root:i=8 residual=0.0011496329680085182
    DEBUG:root:i=9 residual=0.0005811753217130899
    DEBUG:root:i=10 residual=0.00029267038917168975
    DEBUG:root:i=11 residual=0.00014554249355569482
    DEBUG:root:i=12 residual=7.149996963562444e-05
    DEBUG:root:i=13 residual=3.474643381196074e-05
    DEBUG:root:i=14 residual=1.5955227354425006e-05
    DEBUG:root:i=15 residual=6.453929472627351e-06
    DEBUG:root:i=16 residual=2.4474927613482578e-06
    DEBUG:root:i=17 residual=8.115676450870524e-07
    INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
    DEBUG:root:i=0 residual=1.384641170501709
    DEBUG:root:i=1 residual=0.07088145613670349
    DEBUG:root:i=2 residual=0.01882273517549038
    DEBUG:root:i=3 residual=0.006958306767046452
    DEBUG:root:i=4 residual=0.002735827350988984
    DEBUG:root:i=5 residual=0.001034560496918857
    DEBUG:root:i=6 residual=0.0003774636425077915
    DEBUG:root:i=7 residual=0.00013533419405575842
    DEBUG:root:i=8 residual=4.82243049191311e-05
    DEBUG:root:i=9 residual=1.7172726074932143e-05
    DEBUG:root:i=10 residual=6.115362793934764e-06
    DEBUG:root:i=11 residual=2.175059762521414e-06
    DEBUG:root:i=12 residual=7.838124247427913e-07
    INFO:root:rank=0 pagerank=2.8859e-01 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=1 pagerank=2.8859e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8859e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8859e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8859e-01 url=www.lawfareblog.com/topics
    INFO:root:rank=6 pagerank=2.8859e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=7 pagerank=2.8859e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8859e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
    DEBUG:root:i=0 residual=1.2609769105911255
    DEBUG:root:i=1 residual=0.49857109785079956
    DEBUG:root:i=2 residual=0.13418608903884888
    DEBUG:root:i=3 residual=0.0692228302359581
    DEBUG:root:i=4 residual=0.023409781977534294
    DEBUG:root:i=5 residual=0.010187160223722458
    DEBUG:root:i=6 residual=0.004906986374408007
    DEBUG:root:i=7 residual=0.002280212240293622
    DEBUG:root:i=8 residual=0.0010744944447651505
    DEBUG:root:i=9 residual=0.0005251469556242228
    DEBUG:root:i=10 residual=0.00026976881781592965
    DEBUG:root:i=11 residual=0.00014569039922207594
    DEBUG:root:i=12 residual=8.227639773394912e-05
    DEBUG:root:i=13 residual=4.813645136891864e-05
    DEBUG:root:i=14 residual=2.879827115975786e-05
    DEBUG:root:i=15 residual=1.7416465198039077e-05
    DEBUG:root:i=16 residual=1.0551282684900798e-05
    DEBUG:root:i=17 residual=6.383040727087064e-06
    DEBUG:root:i=18 residual=3.8493412830575835e-06
    DEBUG:root:i=19 residual=2.2995980089035584e-06
    DEBUG:root:i=20 residual=1.36830885821837e-06
    DEBUG:root:i=21 residual=8.107048188321642e-07
    INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
    INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
    INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
    DEBUG:root:i=0 residual=1.2827692031860352
    DEBUG:root:i=1 residual=0.5695649981498718
    DEBUG:root:i=2 residual=0.38299471139907837
    DEBUG:root:i=3 residual=0.2173936367034912
    DEBUG:root:i=4 residual=0.140450581908226
    DEBUG:root:i=5 residual=0.1085134968161583
    DEBUG:root:i=6 residual=0.09284141659736633
    DEBUG:root:i=7 residual=0.0822555422782898
    DEBUG:root:i=8 residual=0.07338894158601761
    DEBUG:root:i=9 residual=0.06561234593391418
    DEBUG:root:i=10 residual=0.05909651890397072
    DEBUG:root:i=11 residual=0.05417545139789581
    DEBUG:root:i=12 residual=0.05111692100763321
    DEBUG:root:i=13 residual=0.04999381676316261
    DEBUG:root:i=14 residual=0.05060894042253494
    DEBUG:root:i=15 residual=0.05252622067928314
    DEBUG:root:i=16 residual=0.05518876761198044
    DEBUG:root:i=17 residual=0.058038532733917236
    DEBUG:root:i=18 residual=0.06059236824512482
    DEBUG:root:i=19 residual=0.062478452920913696
    DEBUG:root:i=20 residual=0.06345325708389282
    DEBUG:root:i=21 residual=0.06340521574020386
    DEBUG:root:i=22 residual=0.06234563887119293
    DEBUG:root:i=23 residual=0.06038374453783035
    DEBUG:root:i=24 residual=0.05769389867782593
    DEBUG:root:i=25 residual=0.05447973310947418
    DEBUG:root:i=26 residual=0.050942786037921906
    DEBUG:root:i=27 residual=0.04726115241646767
    DEBUG:root:i=28 residual=0.04357859864830971
    DEBUG:root:i=29 residual=0.040001627057790756
    DEBUG:root:i=30 residual=0.03660229220986366
    DEBUG:root:i=31 residual=0.03342435136437416
    DEBUG:root:i=32 residual=0.030489366501569748
    DEBUG:root:i=33 residual=0.027803201228380203
    DEBUG:root:i=34 residual=0.02536066807806492
    DEBUG:root:i=35 residual=0.023149998858571053
    DEBUG:root:i=36 residual=0.021155372262001038
    DEBUG:root:i=37 residual=0.019359204918146133
    DEBUG:root:i=38 residual=0.017743581905961037
    DEBUG:root:i=39 residual=0.0162908136844635
    DEBUG:root:i=40 residual=0.01498429849743843
    DEBUG:root:i=41 residual=0.013808625750243664
    DEBUG:root:i=42 residual=0.012749756686389446
    DEBUG:root:i=43 residual=0.011794951744377613
    DEBUG:root:i=44 residual=0.01093289628624916
    DEBUG:root:i=45 residual=0.010153383016586304
    DEBUG:root:i=46 residual=0.00944743026047945
    DEBUG:root:i=47 residual=0.008807026781141758
    DEBUG:root:i=48 residual=0.008225115947425365
    DEBUG:root:i=49 residual=0.007695477455854416
    DEBUG:root:i=50 residual=0.0072125839069485664
    DEBUG:root:i=51 residual=0.006771519314497709
    DEBUG:root:i=52 residual=0.006367966067045927
    DEBUG:root:i=53 residual=0.005998097825795412
    DEBUG:root:i=54 residual=0.005658585112541914
    DEBUG:root:i=55 residual=0.005346325691789389
    DEBUG:root:i=56 residual=0.0050587416626513
    DEBUG:root:i=57 residual=0.004793417174369097
    DEBUG:root:i=58 residual=0.004548229742795229
    DEBUG:root:i=59 residual=0.0043213157914578915
    DEBUG:root:i=60 residual=0.004110963083803654
    DEBUG:root:i=61 residual=0.003915724810212851
    DEBUG:root:i=62 residual=0.0037341988645493984
    DEBUG:root:i=63 residual=0.0035651803482323885
    DEBUG:root:i=64 residual=0.0034076140727847815
    DEBUG:root:i=65 residual=0.0032605393789708614
    DEBUG:root:i=66 residual=0.003123006084933877
    DEBUG:root:i=67 residual=0.0029942726250737906
    DEBUG:root:i=68 residual=0.0028736209496855736
    DEBUG:root:i=69 residual=0.002760386560112238
    DEBUG:root:i=70 residual=0.0026539897080510855
    DEBUG:root:i=71 residual=0.0025539074558764696
    DEBUG:root:i=72 residual=0.002459645736962557
    DEBUG:root:i=73 residual=0.002370776142925024
    DEBUG:root:i=74 residual=0.0022868812084198
    DEBUG:root:i=75 residual=0.0022076282184571028
    DEBUG:root:i=76 residual=0.002132629742845893
    DEBUG:root:i=77 residual=0.0020616406109184027
    DEBUG:root:i=78 residual=0.001994348829612136
    DEBUG:root:i=79 residual=0.0019305141177028418
    DEBUG:root:i=80 residual=0.0018698654603213072
    DEBUG:root:i=81 residual=0.001812250236980617
    DEBUG:root:i=82 residual=0.0017574188532307744
    DEBUG:root:i=83 residual=0.0017052083276212215
    DEBUG:root:i=84 residual=0.0016554591711610556
    DEBUG:root:i=85 residual=0.0016080171335488558
    DEBUG:root:i=86 residual=0.0015627103857696056
    DEBUG:root:i=87 residual=0.0015194298466667533
    DEBUG:root:i=88 residual=0.0014780514175072312
    DEBUG:root:i=89 residual=0.0014384748646989465
    DEBUG:root:i=90 residual=0.0014005466364324093
    DEBUG:root:i=91 residual=0.0013642263365909457
    DEBUG:root:i=92 residual=0.0013293909141793847
    DEBUG:root:i=93 residual=0.0012959621381014585
    DEBUG:root:i=94 residual=0.0012638671323657036
    DEBUG:root:i=95 residual=0.0012330348836258054
    DEBUG:root:i=96 residual=0.0012033873936161399
    DEBUG:root:i=97 residual=0.0011748654069378972
    DEBUG:root:i=98 residual=0.0011474041966721416
    DEBUG:root:i=99 residual=0.0011209663935005665
    DEBUG:root:i=100 residual=0.0010954777244478464
    DEBUG:root:i=101 residual=0.0010709073394536972
    DEBUG:root:i=102 residual=0.0010471907444298267
    DEBUG:root:i=103 residual=0.001024293014779687
    DEBUG:root:i=104 residual=0.0010021794587373734
    DEBUG:root:i=105 residual=0.000980809098109603
    DEBUG:root:i=106 residual=0.0009601540514267981
    DEBUG:root:i=107 residual=0.0009401614661328495
    DEBUG:root:i=108 residual=0.000920827267691493
    DEBUG:root:i=109 residual=0.0009020882425829768
    DEBUG:root:i=110 residual=0.0008839504444040358
    DEBUG:root:i=111 residual=0.0008663705084472895
    DEBUG:root:i=112 residual=0.0008493200875818729
    DEBUG:root:i=113 residual=0.0008327661780640483
    DEBUG:root:i=114 residual=0.0008167159976437688
    DEBUG:root:i=115 residual=0.0008011450991034508
    DEBUG:root:i=116 residual=0.0007860048208385706
    DEBUG:root:i=117 residual=0.0007713011000305414
    DEBUG:root:i=118 residual=0.0007570073939859867
    DEBUG:root:i=119 residual=0.0007431143894791603
    DEBUG:root:i=120 residual=0.0007295939140021801
    DEBUG:root:i=121 residual=0.0007164477137848735
    DEBUG:root:i=122 residual=0.0007036403403617442
    DEBUG:root:i=123 residual=0.000691176566760987
    DEBUG:root:i=124 residual=0.0006790340412408113
    DEBUG:root:i=125 residual=0.0006671996088698506
    DEBUG:root:i=126 residual=0.0006556707085110247
    DEBUG:root:i=127 residual=0.0006444298196583986
    DEBUG:root:i=128 residual=0.0006334534264169633
    DEBUG:root:i=129 residual=0.0006227639969438314
    DEBUG:root:i=130 residual=0.0006123227649368346
    DEBUG:root:i=131 residual=0.0006021389272063971
    DEBUG:root:i=132 residual=0.000592191587202251
    DEBUG:root:i=133 residual=0.0005824642139486969
    DEBUG:root:i=134 residual=0.0005729799740947783
    DEBUG:root:i=135 residual=0.0005637028953060508
    DEBUG:root:i=136 residual=0.0005546383908949792
    DEBUG:root:i=137 residual=0.0005457771476358175
    DEBUG:root:i=138 residual=0.0005371171864680946
    DEBUG:root:i=139 residual=0.0005286390078254044
    DEBUG:root:i=140 residual=0.0005203487235121429
    DEBUG:root:i=141 residual=0.0005122360889799893
    DEBUG:root:i=142 residual=0.0005042948178015649
    DEBUG:root:i=143 residual=0.0004965206026099622
    DEBUG:root:i=144 residual=0.0004889043048024178
    DEBUG:root:i=145 residual=0.0004814529966097325
    DEBUG:root:i=146 residual=0.00047415337758138776
    DEBUG:root:i=147 residual=0.00046699715312570333
    DEBUG:root:i=148 residual=0.0004599883686751127
    DEBUG:root:i=149 residual=0.00045312196016311646
    DEBUG:root:i=150 residual=0.0004463903314899653
    DEBUG:root:i=151 residual=0.00043978862231597304
    DEBUG:root:i=152 residual=0.00043331709457561374
    DEBUG:root:i=153 residual=0.0004269649798516184
    DEBUG:root:i=154 residual=0.0004207403399050236
    DEBUG:root:i=155 residual=0.000414626847486943
    DEBUG:root:i=156 residual=0.0004086379485670477
    DEBUG:root:i=157 residual=0.0004027560062240809
    DEBUG:root:i=158 residual=0.0003969801473431289
    DEBUG:root:i=159 residual=0.0003913167747668922
    DEBUG:root:i=160 residual=0.00038575270446017385
    DEBUG:root:i=161 residual=0.0003802915452979505
    DEBUG:root:i=162 residual=0.0003749225288629532
    DEBUG:root:i=163 residual=0.0003696526400744915
    DEBUG:root:i=164 residual=0.0003644817625172436
    DEBUG:root:i=165 residual=0.0003593922592699528
    DEBUG:root:i=166 residual=0.0003543993807397783
    DEBUG:root:i=167 residual=0.0003494885459076613
    DEBUG:root:i=168 residual=0.0003446690388955176
    DEBUG:root:i=169 residual=0.00033992581302300096
    DEBUG:root:i=170 residual=0.0003352679777890444
    DEBUG:root:i=171 residual=0.00033068869379349053
    DEBUG:root:i=172 residual=0.0003261821111664176
    DEBUG:root:i=173 residual=0.00032175626256503165
    DEBUG:root:i=174 residual=0.00031739857513457537
    DEBUG:root:i=175 residual=0.0003131199337076396
    DEBUG:root:i=176 residual=0.0003089089004788548
    DEBUG:root:i=177 residual=0.0003047662612516433
    DEBUG:root:i=178 residual=0.0003006916376762092
    DEBUG:root:i=179 residual=0.0002966824104078114
    DEBUG:root:i=180 residual=0.00029273840482346714
    DEBUG:root:i=181 residual=0.0002888537710532546
    DEBUG:root:i=182 residual=0.0002850363962352276
    DEBUG:root:i=183 residual=0.0002812776656355709
    DEBUG:root:i=184 residual=0.00027757816133089364
    DEBUG:root:i=185 residual=0.0002739362826105207
    DEBUG:root:i=186 residual=0.0002703509817365557
    DEBUG:root:i=187 residual=0.0002668218221515417
    DEBUG:root:i=188 residual=0.0002633473486639559
    DEBUG:root:i=189 residual=0.0002599271247163415
    DEBUG:root:i=190 residual=0.0002565589384175837
    DEBUG:root:i=191 residual=0.00025324473972432315
    DEBUG:root:i=192 residual=0.0002499780966900289
    DEBUG:root:i=193 residual=0.00024675921304151416
    DEBUG:root:i=194 residual=0.00024359016970265657
    DEBUG:root:i=195 residual=0.00024046747421380132
    DEBUG:root:i=196 residual=0.00023739534663036466
    DEBUG:root:i=197 residual=0.00023436312039848417
    DEBUG:root:i=198 residual=0.0002313829172635451
    DEBUG:root:i=199 residual=0.00022844377963338047
    DEBUG:root:i=200 residual=0.00022554656607098877
    DEBUG:root:i=201 residual=0.00022269324108492583
    DEBUG:root:i=202 residual=0.0002198815782321617
    DEBUG:root:i=203 residual=0.00021710718283429742
    DEBUG:root:i=204 residual=0.0002143743768101558
    DEBUG:root:i=205 residual=0.00021168560488149524
    DEBUG:root:i=206 residual=0.00020903248514514416
    DEBUG:root:i=207 residual=0.00020641411538235843
    DEBUG:root:i=208 residual=0.0002038391394307837
    DEBUG:root:i=209 residual=0.0002012984623434022
    DEBUG:root:i=210 residual=0.00019879452884197235
    DEBUG:root:i=211 residual=0.00019632479234132916
    DEBUG:root:i=212 residual=0.00019389066437724978
    DEBUG:root:i=213 residual=0.00019148817227687687
    DEBUG:root:i=214 residual=0.00018912232189904898
    DEBUG:root:i=215 residual=0.00018678708875086159
    DEBUG:root:i=216 residual=0.0001844863290898502
    DEBUG:root:i=217 residual=0.00018221736536361277
    DEBUG:root:i=218 residual=0.0001799802266759798
    DEBUG:root:i=219 residual=0.00017777214816305786
    DEBUG:root:i=220 residual=0.00017559781554155052
    DEBUG:root:i=221 residual=0.0001734526304062456
    DEBUG:root:i=222 residual=0.00017133493383880705
    DEBUG:root:i=223 residual=0.0001692432997515425
    DEBUG:root:i=224 residual=0.00016718725964892656
    DEBUG:root:i=225 residual=0.0001651525526540354
    DEBUG:root:i=226 residual=0.00016314884123858064
    DEBUG:root:i=227 residual=0.00016117095947265625
    DEBUG:root:i=228 residual=0.00015921950398478657
    DEBUG:root:i=229 residual=0.00015729738515801728
    DEBUG:root:i=230 residual=0.0001553953770780936
    DEBUG:root:i=231 residual=0.00015352330228779465
    DEBUG:root:i=232 residual=0.0001516754855401814
    DEBUG:root:i=233 residual=0.0001498528872616589
    DEBUG:root:i=234 residual=0.00014805006503593177
    DEBUG:root:i=235 residual=0.00014627512427978218
    DEBUG:root:i=236 residual=0.0001445232774131
    DEBUG:root:i=237 residual=0.00014279421884566545
    DEBUG:root:i=238 residual=0.0001410877302987501
    DEBUG:root:i=239 residual=0.00013940170174464583
    DEBUG:root:i=240 residual=0.00013774022227153182
    DEBUG:root:i=241 residual=0.00013609939196612686
    DEBUG:root:i=242 residual=0.00013447874516714364
    DEBUG:root:i=243 residual=0.00013288078480400145
    DEBUG:root:i=244 residual=0.00013130540901329368
    DEBUG:root:i=245 residual=0.00012974666606169194
    DEBUG:root:i=246 residual=0.00012820954725611955
    DEBUG:root:i=247 residual=0.00012669457646552473
    DEBUG:root:i=248 residual=0.00012519383744802326
    DEBUG:root:i=249 residual=0.00012371821503620595
    DEBUG:root:i=250 residual=0.00012225793034303933
    DEBUG:root:i=251 residual=0.00012081572640454397
    DEBUG:root:i=252 residual=0.00011939577962039039
    DEBUG:root:i=253 residual=0.00011799107596743852
    DEBUG:root:i=254 residual=0.00011660430755000561
    DEBUG:root:i=255 residual=0.00011523547436809167
    DEBUG:root:i=256 residual=0.00011388534039724618
    DEBUG:root:i=257 residual=0.0001125517301261425
    DEBUG:root:i=258 residual=0.00011123301374027506
    DEBUG:root:i=259 residual=0.00010993094474542886
    DEBUG:root:i=260 residual=0.00010864681098610163
    DEBUG:root:i=261 residual=0.00010737804404925555
    DEBUG:root:i=262 residual=0.00010612538608256727
    DEBUG:root:i=263 residual=0.00010488829866517335
    DEBUG:root:i=264 residual=0.00010366611240897328
    DEBUG:root:i=265 residual=0.00010246025340165943
    DEBUG:root:i=266 residual=0.00010126799315912649
    DEBUG:root:i=267 residual=0.00010009056859416887
    DEBUG:root:i=268 residual=9.892947127809748e-05
    DEBUG:root:i=269 residual=9.778169624041766e-05
    DEBUG:root:i=270 residual=9.664885874371976e-05
    DEBUG:root:i=271 residual=9.553002018947154e-05
    DEBUG:root:i=272 residual=9.442441660212353e-05
    DEBUG:root:i=273 residual=9.333471825812012e-05
    DEBUG:root:i=274 residual=9.225399844581261e-05
    DEBUG:root:i=275 residual=9.119057358475402e-05
    DEBUG:root:i=276 residual=9.013769886223599e-05
    DEBUG:root:i=277 residual=8.910186443245038e-05
    DEBUG:root:i=278 residual=8.807366248220205e-05
    DEBUG:root:i=279 residual=8.70603762450628e-05
    DEBUG:root:i=280 residual=8.606079063611105e-05
    DEBUG:root:i=281 residual=8.507032180204988e-05
    DEBUG:root:i=282 residual=8.409340080106631e-05
    DEBUG:root:i=283 residual=8.312865247717127e-05
    DEBUG:root:i=284 residual=8.217369031626731e-05
    DEBUG:root:i=285 residual=8.123160660034046e-05
    DEBUG:root:i=286 residual=8.030069147935137e-05
    DEBUG:root:i=287 residual=7.938373164506629e-05
    DEBUG:root:i=288 residual=7.8474564361386e-05
    DEBUG:root:i=289 residual=7.757738057989627e-05
    DEBUG:root:i=290 residual=7.66910525271669e-05
    DEBUG:root:i=291 residual=7.581434329040349e-05
    DEBUG:root:i=292 residual=7.495108002331108e-05
    DEBUG:root:i=293 residual=7.409568934235722e-05
    DEBUG:root:i=294 residual=7.32520129531622e-05
    DEBUG:root:i=295 residual=7.241595449158922e-05
    DEBUG:root:i=296 residual=7.15929563739337e-05
    DEBUG:root:i=297 residual=7.077780173858628e-05
    DEBUG:root:i=298 residual=6.997357559157535e-05
    DEBUG:root:i=299 residual=6.917796417837963e-05
    DEBUG:root:i=300 residual=6.839350680820644e-05
    DEBUG:root:i=301 residual=6.761519034625962e-05
    DEBUG:root:i=302 residual=6.684820255031809e-05
    DEBUG:root:i=303 residual=6.608971307286993e-05
    DEBUG:root:i=304 residual=6.534148269565776e-05
    DEBUG:root:i=305 residual=6.460233998950571e-05
    DEBUG:root:i=306 residual=6.386945460690185e-05
    DEBUG:root:i=307 residual=6.314725033007562e-05
    DEBUG:root:i=308 residual=6.24332606093958e-05
    DEBUG:root:i=309 residual=6.17267214693129e-05
    DEBUG:root:i=310 residual=6.102882616687566e-05
    DEBUG:root:i=311 residual=6.033850149833597e-05
    DEBUG:root:i=312 residual=5.965821765130386e-05
    DEBUG:root:i=313 residual=5.898446397623047e-05
    DEBUG:root:i=314 residual=5.8319157687947154e-05
    DEBUG:root:i=315 residual=5.76618876948487e-05
    DEBUG:root:i=316 residual=5.700935798813589e-05
    DEBUG:root:i=317 residual=5.636995047098026e-05
    DEBUG:root:i=318 residual=5.573379166889936e-05
    DEBUG:root:i=319 residual=5.510630217031576e-05
    DEBUG:root:i=320 residual=5.4487270972458646e-05
    DEBUG:root:i=321 residual=5.387392229749821e-05
    DEBUG:root:i=322 residual=5.326850077835843e-05
    DEBUG:root:i=323 residual=5.2671428420580924e-05
    DEBUG:root:i=324 residual=5.207939102547243e-05
    DEBUG:root:i=325 residual=5.1494993385858834e-05
    DEBUG:root:i=326 residual=5.0917427870444953e-05
    DEBUG:root:i=327 residual=5.034517380408943e-05
    DEBUG:root:i=328 residual=4.978201468475163e-05
    DEBUG:root:i=329 residual=4.922328662360087e-05
    DEBUG:root:i=330 residual=4.867058669333346e-05
    DEBUG:root:i=331 residual=4.812569386558607e-05
    DEBUG:root:i=332 residual=4.75873748655431e-05
    DEBUG:root:i=333 residual=4.70541235699784e-05
    DEBUG:root:i=334 residual=4.652840652852319e-05
    DEBUG:root:i=335 residual=4.600807005772367e-05
    DEBUG:root:i=336 residual=4.5493583456845954e-05
    DEBUG:root:i=337 residual=4.498632188187912e-05
    DEBUG:root:i=338 residual=4.4483229430625215e-05
    DEBUG:root:i=339 residual=4.398632881930098e-05
    DEBUG:root:i=340 residual=4.3495307181729004e-05
    DEBUG:root:i=341 residual=4.301048829802312e-05
    DEBUG:root:i=342 residual=4.253100632922724e-05
    DEBUG:root:i=343 residual=4.205688674119301e-05
    DEBUG:root:i=344 residual=4.158800220466219e-05
    DEBUG:root:i=345 residual=4.112588067073375e-05
    DEBUG:root:i=346 residual=4.066789915668778e-05
    DEBUG:root:i=347 residual=4.0215341869043186e-05
    DEBUG:root:i=348 residual=3.9766659028828144e-05
    DEBUG:root:i=349 residual=3.932389154215343e-05
    DEBUG:root:i=350 residual=3.88877306249924e-05
    DEBUG:root:i=351 residual=3.845495666610077e-05
    DEBUG:root:i=352 residual=3.802768696914427e-05
    DEBUG:root:i=353 residual=3.760487379622646e-05
    DEBUG:root:i=354 residual=3.7186811823630705e-05
    DEBUG:root:i=355 residual=3.67724169336725e-05
    DEBUG:root:i=356 residual=3.6365883715916425e-05
    DEBUG:root:i=357 residual=3.5960940294899046e-05
    DEBUG:root:i=358 residual=3.556324008968659e-05
    DEBUG:root:i=359 residual=3.51675771526061e-05
    DEBUG:root:i=360 residual=3.47784734913148e-05
    DEBUG:root:i=361 residual=3.439195279497653e-05
    DEBUG:root:i=362 residual=3.401129288249649e-05
    DEBUG:root:i=363 residual=3.3633539715083316e-05
    DEBUG:root:i=364 residual=3.326184014440514e-05
    DEBUG:root:i=365 residual=3.2894982723519206e-05
    DEBUG:root:i=366 residual=3.2530344469705597e-05
    DEBUG:root:i=367 residual=3.216938421246596e-05
    DEBUG:root:i=368 residual=3.1814244721317664e-05
    DEBUG:root:i=369 residual=3.146123344777152e-05
    DEBUG:root:i=370 residual=3.111379555775784e-05
    DEBUG:root:i=371 residual=3.077023211517371e-05
    DEBUG:root:i=372 residual=3.0429238904616795e-05
    DEBUG:root:i=373 residual=3.009220199601259e-05
    DEBUG:root:i=374 residual=2.9760843972326256e-05
    DEBUG:root:i=375 residual=2.9431477742036805e-05
    DEBUG:root:i=376 residual=2.9106036890880205e-05
    DEBUG:root:i=377 residual=2.878546183637809e-05
    DEBUG:root:i=378 residual=2.8467204174376093e-05
    DEBUG:root:i=379 residual=2.8153726816526614e-05
    DEBUG:root:i=380 residual=2.7843450880027376e-05
    DEBUG:root:i=381 residual=2.7536420020624064e-05
    DEBUG:root:i=382 residual=2.7231335479882546e-05
    DEBUG:root:i=383 residual=2.6932768378173932e-05
    DEBUG:root:i=384 residual=2.6633464585756883e-05
    DEBUG:root:i=385 residual=2.6341998818679713e-05
    DEBUG:root:i=386 residual=2.6049934604088776e-05
    DEBUG:root:i=387 residual=2.5764387828530744e-05
    DEBUG:root:i=388 residual=2.5479452233412303e-05
    DEBUG:root:i=389 residual=2.520026100683026e-05
    DEBUG:root:i=390 residual=2.4921622753026895e-05
    DEBUG:root:i=391 residual=2.464789940859191e-05
    DEBUG:root:i=392 residual=2.4378037778660655e-05
    DEBUG:root:i=393 residual=2.410839260846842e-05
    DEBUG:root:i=394 residual=2.38443972193636e-05
    DEBUG:root:i=395 residual=2.358129495405592e-05
    DEBUG:root:i=396 residual=2.33220198424533e-05
    DEBUG:root:i=397 residual=2.306620626768563e-05
    DEBUG:root:i=398 residual=2.2812808310845867e-05
    DEBUG:root:i=399 residual=2.256167499581352e-05
    DEBUG:root:i=400 residual=2.2313022782327607e-05
    DEBUG:root:i=401 residual=2.2068643374950625e-05
    DEBUG:root:i=402 residual=2.182490425184369e-05
    DEBUG:root:i=403 residual=2.1586365619441494e-05
    DEBUG:root:i=404 residual=2.134922397090122e-05
    DEBUG:root:i=405 residual=2.1115349227329716e-05
    DEBUG:root:i=406 residual=2.0881341697531752e-05
    DEBUG:root:i=407 residual=2.065471198875457e-05
    DEBUG:root:i=408 residual=2.0426336050149985e-05
    DEBUG:root:i=409 residual=2.0203035091981292e-05
    DEBUG:root:i=410 residual=1.998182779061608e-05
    DEBUG:root:i=411 residual=1.9762574083870277e-05
    DEBUG:root:i=412 residual=1.9545408576959744e-05
    DEBUG:root:i=413 residual=1.9330505892867222e-05
    DEBUG:root:i=414 residual=1.9119715943816118e-05
    DEBUG:root:i=415 residual=1.89092788787093e-05
    DEBUG:root:i=416 residual=1.870133201009594e-05
    DEBUG:root:i=417 residual=1.8496173652238213e-05
    DEBUG:root:i=418 residual=1.8294178516953252e-05
    DEBUG:root:i=419 residual=1.8092923710355535e-05
    DEBUG:root:i=420 residual=1.789570160326548e-05
    DEBUG:root:i=421 residual=1.769969276210759e-05
    DEBUG:root:i=422 residual=1.7506210497231223e-05
    DEBUG:root:i=423 residual=1.731361226120498e-05
    DEBUG:root:i=424 residual=1.7126019884017296e-05
    DEBUG:root:i=425 residual=1.6936224710661918e-05
    DEBUG:root:i=426 residual=1.6752186638768762e-05
    DEBUG:root:i=427 residual=1.6568301361985505e-05
    DEBUG:root:i=428 residual=1.6387832147302106e-05
    DEBUG:root:i=429 residual=1.62082833412569e-05
    DEBUG:root:i=430 residual=1.603088094270788e-05
    DEBUG:root:i=431 residual=1.5855972378631122e-05
    DEBUG:root:i=432 residual=1.568248262628913e-05
    DEBUG:root:i=433 residual=1.5510322555201128e-05
    DEBUG:root:i=434 residual=1.5340798199758865e-05
    DEBUG:root:i=435 residual=1.5173813153523952e-05
    DEBUG:root:i=436 residual=1.5007540241640527e-05
    DEBUG:root:i=437 residual=1.4845551959297154e-05
    DEBUG:root:i=438 residual=1.4680691492685582e-05
    DEBUG:root:i=439 residual=1.4520471268042456e-05
    DEBUG:root:i=440 residual=1.4363050468091387e-05
    DEBUG:root:i=441 residual=1.4205108527676202e-05
    DEBUG:root:i=442 residual=1.4048922821530141e-05
    DEBUG:root:i=443 residual=1.3894798030378297e-05
    DEBUG:root:i=444 residual=1.3744444004260004e-05
    DEBUG:root:i=445 residual=1.3593024959845934e-05
    DEBUG:root:i=446 residual=1.3445413060253486e-05
    DEBUG:root:i=447 residual=1.3299808415467851e-05
    DEBUG:root:i=448 residual=1.3152850442565978e-05
    DEBUG:root:i=449 residual=1.3010997463425156e-05
    DEBUG:root:i=450 residual=1.2867431905760895e-05
    DEBUG:root:i=451 residual=1.272639292437816e-05
    DEBUG:root:i=452 residual=1.258912652701838e-05
    DEBUG:root:i=453 residual=1.2450375834305305e-05
    DEBUG:root:i=454 residual=1.23142253869446e-05
    DEBUG:root:i=455 residual=1.2182147656858433e-05
    DEBUG:root:i=456 residual=1.2048786629748065e-05
    DEBUG:root:i=457 residual=1.19169144454645e-05
    DEBUG:root:i=458 residual=1.1786254617618397e-05
    DEBUG:root:i=459 residual=1.1656698006845545e-05
    DEBUG:root:i=460 residual=1.153135690401541e-05
    DEBUG:root:i=461 residual=1.1405291843402665e-05
    DEBUG:root:i=462 residual=1.1280823855486233e-05
    DEBUG:root:i=463 residual=1.1157427252328489e-05
    DEBUG:root:i=464 residual=1.1035820534743834e-05
    DEBUG:root:i=465 residual=1.0916379324044101e-05
    DEBUG:root:i=466 residual=1.0797417417052202e-05
    DEBUG:root:i=467 residual=1.0678880244086031e-05
    DEBUG:root:i=468 residual=1.0562967872829176e-05
    DEBUG:root:i=469 residual=1.0446318810863886e-05
    DEBUG:root:i=470 residual=1.0333634236303624e-05
    DEBUG:root:i=471 residual=1.0221146112598944e-05
    DEBUG:root:i=472 residual=1.0110405128216371e-05
    DEBUG:root:i=473 residual=9.999658686865587e-06
    DEBUG:root:i=474 residual=9.891376066661905e-06
    DEBUG:root:i=475 residual=9.781921107787639e-06
    DEBUG:root:i=476 residual=9.676406989456154e-06
    DEBUG:root:i=477 residual=9.571065675118007e-06
    DEBUG:root:i=478 residual=9.467221389058977e-06
    DEBUG:root:i=479 residual=9.362394848722033e-06
    DEBUG:root:i=480 residual=9.261209925170988e-06
    DEBUG:root:i=481 residual=9.162384230876341e-06
    DEBUG:root:i=482 residual=9.060299817065243e-06
    DEBUG:root:i=483 residual=8.963802429207135e-06
    DEBUG:root:i=484 residual=8.86373072717106e-06
    DEBUG:root:i=485 residual=8.769355190452188e-06
    DEBUG:root:i=486 residual=8.6746631495771e-06
    DEBUG:root:i=487 residual=8.578858796681743e-06
    DEBUG:root:i=488 residual=8.485506441502366e-06
    DEBUG:root:i=489 residual=8.395071745326277e-06
    DEBUG:root:i=490 residual=8.302647074742708e-06
    DEBUG:root:i=491 residual=8.211555723391939e-06
    DEBUG:root:i=492 residual=8.121851351461373e-06
    DEBUG:root:i=493 residual=8.034638085518964e-06
    DEBUG:root:i=494 residual=7.946824553073384e-06
    DEBUG:root:i=495 residual=7.860609002818819e-06
    DEBUG:root:i=496 residual=7.776722668495495e-06
    DEBUG:root:i=497 residual=7.690749953326304e-06
    DEBUG:root:i=498 residual=7.6080600592831615e-06
    DEBUG:root:i=499 residual=7.525772616645554e-06
    DEBUG:root:i=500 residual=7.4441704782657325e-06
    DEBUG:root:i=501 residual=7.363049462583149e-06
    DEBUG:root:i=502 residual=7.282740625669248e-06
    DEBUG:root:i=503 residual=7.2027701207844075e-06
    DEBUG:root:i=504 residual=7.126005584723316e-06
    DEBUG:root:i=505 residual=7.047215149214026e-06
    DEBUG:root:i=506 residual=6.970124559302349e-06
    DEBUG:root:i=507 residual=6.8962390287197195e-06
    DEBUG:root:i=508 residual=6.819588634243701e-06
    DEBUG:root:i=509 residual=6.745581231371034e-06
    DEBUG:root:i=510 residual=6.672919880656991e-06
    DEBUG:root:i=511 residual=6.599707830901025e-06
    DEBUG:root:i=512 residual=6.528746325784596e-06
    DEBUG:root:i=513 residual=6.456593382608844e-06
    DEBUG:root:i=514 residual=6.3873712861095555e-06
    DEBUG:root:i=515 residual=6.317122370091965e-06
    DEBUG:root:i=516 residual=6.249343641684391e-06
    DEBUG:root:i=517 residual=6.181364369695075e-06
    DEBUG:root:i=518 residual=6.113037670729682e-06
    DEBUG:root:i=519 residual=6.04914384894073e-06
    DEBUG:root:i=520 residual=5.982505172141828e-06
    DEBUG:root:i=521 residual=5.918391707382398e-06
    DEBUG:root:i=522 residual=5.853769380337326e-06
    DEBUG:root:i=523 residual=5.790087016066536e-06
    DEBUG:root:i=524 residual=5.7279880820715334e-06
    DEBUG:root:i=525 residual=5.665828211931512e-06
    DEBUG:root:i=526 residual=5.603707904811017e-06
    DEBUG:root:i=527 residual=5.5426894505217206e-06
    DEBUG:root:i=528 residual=5.481562311615562e-06
    DEBUG:root:i=529 residual=5.422944923338946e-06
    DEBUG:root:i=530 residual=5.363569925975753e-06
    DEBUG:root:i=531 residual=5.305489594320534e-06
    DEBUG:root:i=532 residual=5.2475179472821765e-06
    DEBUG:root:i=533 residual=5.189071089262143e-06
    DEBUG:root:i=534 residual=5.134525054018013e-06
    DEBUG:root:i=535 residual=5.0786056817742065e-06
    DEBUG:root:i=536 residual=5.024844540457707e-06
    DEBUG:root:i=537 residual=4.970526788383722e-06
    DEBUG:root:i=538 residual=4.915289537166245e-06
    DEBUG:root:i=539 residual=4.861879006057279e-06
    DEBUG:root:i=540 residual=4.8079623411467765e-06
    DEBUG:root:i=541 residual=4.758530849358067e-06
    DEBUG:root:i=542 residual=4.706941581389401e-06
    DEBUG:root:i=543 residual=4.657149020204088e-06
    DEBUG:root:i=544 residual=4.604737569025019e-06
    DEBUG:root:i=545 residual=4.555153736873763e-06
    DEBUG:root:i=546 residual=4.50600282420055e-06
    DEBUG:root:i=547 residual=4.456157057575183e-06
    DEBUG:root:i=548 residual=4.407340384204872e-06
    DEBUG:root:i=549 residual=4.360890216048574e-06
    DEBUG:root:i=550 residual=4.312411419959972e-06
    DEBUG:root:i=551 residual=4.266420546628069e-06
    DEBUG:root:i=552 residual=4.220827122480841e-06
    DEBUG:root:i=553 residual=4.173604793322738e-06
    DEBUG:root:i=554 residual=4.1327093640575185e-06
    DEBUG:root:i=555 residual=4.085788987140404e-06
    DEBUG:root:i=556 residual=4.040225576318335e-06
    DEBUG:root:i=557 residual=3.996608029410709e-06
    DEBUG:root:i=558 residual=3.955151896661846e-06
    DEBUG:root:i=559 residual=3.909524366463302e-06
    DEBUG:root:i=560 residual=3.867378381983144e-06
    DEBUG:root:i=561 residual=3.824963187071262e-06
    DEBUG:root:i=562 residual=3.7839995457034092e-06
    DEBUG:root:i=563 residual=3.7417401017592056e-06
    DEBUG:root:i=564 residual=3.7020686249888968e-06
    DEBUG:root:i=565 residual=3.661782102426514e-06
    DEBUG:root:i=566 residual=3.6244648526917445e-06
    DEBUG:root:i=567 residual=3.584880232665455e-06
    DEBUG:root:i=568 residual=3.5445295907265972e-06
    DEBUG:root:i=569 residual=3.5064135772699956e-06
    DEBUG:root:i=570 residual=3.4660843084566295e-06
    DEBUG:root:i=571 residual=3.431974846535013e-06
    DEBUG:root:i=572 residual=3.3939213608391583e-06
    DEBUG:root:i=573 residual=3.357022251293529e-06
    DEBUG:root:i=574 residual=3.3200899451912846e-06
    DEBUG:root:i=575 residual=3.2841055599419633e-06
    DEBUG:root:i=576 residual=3.249057954235468e-06
    DEBUG:root:i=577 residual=3.2129928513313644e-06
    DEBUG:root:i=578 residual=3.178400675096782e-06
    DEBUG:root:i=579 residual=3.1440818020200823e-06
    DEBUG:root:i=580 residual=3.1091165055840975e-06
    DEBUG:root:i=581 residual=3.0764324492338346e-06
    DEBUG:root:i=582 residual=3.0429894195549423e-06
    DEBUG:root:i=583 residual=3.0107341899565654e-06
    DEBUG:root:i=584 residual=2.9776936116832076e-06
    DEBUG:root:i=585 residual=2.95007043860096e-06
    DEBUG:root:i=586 residual=2.9183479455241468e-06
    DEBUG:root:i=587 residual=2.8833771921199514e-06
    DEBUG:root:i=588 residual=2.851646058843471e-06
    DEBUG:root:i=589 residual=2.8206088700244436e-06
    DEBUG:root:i=590 residual=2.7885680538020097e-06
    DEBUG:root:i=591 residual=2.760008101176936e-06
    DEBUG:root:i=592 residual=2.7292257982480805e-06
    DEBUG:root:i=593 residual=2.7011355996364728e-06
    DEBUG:root:i=594 residual=2.6706140943133505e-06
    DEBUG:root:i=595 residual=2.6415464162710123e-06
    DEBUG:root:i=596 residual=2.613428250697325e-06
    DEBUG:root:i=597 residual=2.584780304459855e-06
    DEBUG:root:i=598 residual=2.5588146854715887e-06
    DEBUG:root:i=599 residual=2.530889787522028e-06
    DEBUG:root:i=600 residual=2.500986965969787e-06
    DEBUG:root:i=601 residual=2.474996335877222e-06
    DEBUG:root:i=602 residual=2.4480552838213043e-06
    DEBUG:root:i=603 residual=2.4217006284743547e-06
    DEBUG:root:i=604 residual=2.3944669464981416e-06
    DEBUG:root:i=605 residual=2.3680099729972426e-06
    DEBUG:root:i=606 residual=2.3425118342856877e-06
    DEBUG:root:i=607 residual=2.32232946473232e-06
    DEBUG:root:i=608 residual=2.2942840587347746e-06
    DEBUG:root:i=609 residual=2.26831934924121e-06
    DEBUG:root:i=610 residual=2.2451692984759575e-06
    DEBUG:root:i=611 residual=2.224397576355841e-06
    DEBUG:root:i=612 residual=2.196592049585888e-06
    DEBUG:root:i=613 residual=2.1720722997997655e-06
    DEBUG:root:i=614 residual=2.1486589503183495e-06
    DEBUG:root:i=615 residual=2.1253670183796203e-06
    DEBUG:root:i=616 residual=2.1036810267105466e-06
    DEBUG:root:i=617 residual=2.0794566353288246e-06
    DEBUG:root:i=618 residual=2.0546990526781883e-06
    DEBUG:root:i=619 residual=2.0344559743534774e-06
    DEBUG:root:i=620 residual=2.012735194512061e-06
    DEBUG:root:i=621 residual=1.9920362319680862e-06
    DEBUG:root:i=622 residual=1.9686656287376536e-06
    DEBUG:root:i=623 residual=1.9523811261024093e-06
    DEBUG:root:i=624 residual=1.930266989802476e-06
    DEBUG:root:i=625 residual=1.909303591673961e-06
    DEBUG:root:i=626 residual=1.8890842738983338e-06
    DEBUG:root:i=627 residual=1.8689986518438673e-06
    DEBUG:root:i=628 residual=1.844037228693196e-06
    DEBUG:root:i=629 residual=1.8266899814989301e-06
    DEBUG:root:i=630 residual=1.8052809309665463e-06
    DEBUG:root:i=631 residual=1.7843328805611236e-06
    DEBUG:root:i=632 residual=1.7664473261902458e-06
    DEBUG:root:i=633 residual=1.7463878521084553e-06
    DEBUG:root:i=634 residual=1.7289062270720024e-06
    DEBUG:root:i=635 residual=1.7103011487051845e-06
    DEBUG:root:i=636 residual=1.6905489701457554e-06
    DEBUG:root:i=637 residual=1.6721699012123281e-06
    DEBUG:root:i=638 residual=1.6578101167397108e-06
    DEBUG:root:i=639 residual=1.63896118010598e-06
    DEBUG:root:i=640 residual=1.6190028873097617e-06
    DEBUG:root:i=641 residual=1.6026028788473923e-06
    DEBUG:root:i=642 residual=1.5878270005487138e-06
    DEBUG:root:i=643 residual=1.567196704854723e-06
    DEBUG:root:i=644 residual=1.5504533621424343e-06
    DEBUG:root:i=645 residual=1.5329467260016827e-06
    DEBUG:root:i=646 residual=1.518024305369181e-06
    DEBUG:root:i=647 residual=1.5016564702818869e-06
    DEBUG:root:i=648 residual=1.4839466757621267e-06
    DEBUG:root:i=649 residual=1.4700534620715189e-06
    DEBUG:root:i=650 residual=1.4516486999127665e-06
    DEBUG:root:i=651 residual=1.4361876310431398e-06
    DEBUG:root:i=652 residual=1.422438572262763e-06
    DEBUG:root:i=653 residual=1.4059027080293163e-06
    DEBUG:root:i=654 residual=1.3923869346399442e-06
    DEBUG:root:i=655 residual=1.3791981245958596e-06
    DEBUG:root:i=656 residual=1.3593449921245337e-06
    DEBUG:root:i=657 residual=1.3452959137794096e-06
    DEBUG:root:i=658 residual=1.3324687415661174e-06
    DEBUG:root:i=659 residual=1.3193518952903105e-06
    DEBUG:root:i=660 residual=1.3081352108201827e-06
    DEBUG:root:i=661 residual=1.2897853594040498e-06
    DEBUG:root:i=662 residual=1.2756421483572922e-06
    DEBUG:root:i=663 residual=1.2590903679665644e-06
    DEBUG:root:i=664 residual=1.2464630572139868e-06
    DEBUG:root:i=665 residual=1.2358984804450301e-06
    DEBUG:root:i=666 residual=1.2212224191898713e-06
    DEBUG:root:i=667 residual=1.2086312608516891e-06
    DEBUG:root:i=668 residual=1.1951071883231634e-06
    DEBUG:root:i=669 residual=1.1830118182842853e-06
    DEBUG:root:i=670 residual=1.176452542495099e-06
    DEBUG:root:i=671 residual=1.1574952623050194e-06
    DEBUG:root:i=672 residual=1.1499214451760054e-06
    DEBUG:root:i=673 residual=1.1317625876472448e-06
    DEBUG:root:i=674 residual=1.117744432121981e-06
    DEBUG:root:i=675 residual=1.1055287814087933e-06
    DEBUG:root:i=676 residual=1.0956734968203818e-06
    DEBUG:root:i=677 residual=1.0849090585907106e-06
    DEBUG:root:i=678 residual=1.0736017657109187e-06
    DEBUG:root:i=679 residual=1.06269749267085e-06
    DEBUG:root:i=680 residual=1.0510285619602655e-06
    DEBUG:root:i=681 residual=1.0440669484523823e-06
    DEBUG:root:i=682 residual=1.0312398899259279e-06
    DEBUG:root:i=683 residual=1.0173085911446833e-06
    DEBUG:root:i=684 residual=1.0039417475127266e-06
    DEBUG:root:i=685 residual=9.928252211466315e-07
    INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 pagerank=3.1758e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
    INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
    INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
    INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
    INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   The assignment is worth 8 points.
   1. There are 6 parts to the output above.  (4 in Task1 and 2 in Task2.)
   1. Each part that you get incorrect will result in -2 points.  (But you cannot go negative.)
   1. Another way of phrasing this is that the first 2 parts you complete are not worth any points,
      but each part after that is worth 2 points.
