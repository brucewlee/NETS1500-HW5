Project: Finding the Optimal Materials for Submarine Cable Protectors
Authors: Bruce Lee (50188955), Wilson Hu (62182170)



Description

We model the global cable network as a weighted graph. 
We score each cable by betweenness and natural-disaster risk.
We rank cable protection materials by reliability. 
We assign materials to cables by a rule that mixes all three scores. 
The model shows a large drop in expected outage cost.



Categories Used

Graph and graph algorithms.  
Physical Networks (Internet).  



Work Breakdown

Bruce: scraped cable data, built the graph, calculated betweenness, computed importance/risk/reliability scores, built the Plotly maps, drafted the report.  
Wilson: processed EM-DAT data, helped with risk scores, wrote the optimization, and helped with the report.