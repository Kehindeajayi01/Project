# Project
**evaluate_struture** script provides a way to calculate both the IOU (Intersection Over Union) value and precision of the predicted bounding boxes of table images against the ground truth.

## Symbol Scraper
- Symbol Scraper extracts the locations and attributes associated with **characters**
and graphics in PDF files. Attributes include fonts, line endpoints, line
widths, etc.  Results are returned in XML, or as an annotated version of an input
PDF file showing the bounding box locations of identified text and graphics.

## Installation Types
Symbol Scraper can be run two ways:
  - From the command line, and
  - through a web service (port 7002 by default)

## Building the Command Line Program 
- To build the program for use at the command line, 
  - first, clone this directory: https://gitlab.com/dprl/symbolscraper-server.git 
  - Then move to the base directory (symbolscraper-server), and then build the system with maven, using the following command (tested on linux systems with java 11): mvn clean package
  - The program now resides in a shaded jar (i.e., a compiled, linked set of jar files) located at target/symbolscraper-server-1.0-SNAPSHOT-shaded.jar.

## Command Line Execution
- To extract the character and page number locations, run the following command:
  - java -jar target/symbolscraper-server-1.0-SNAPSHOT-shaded.jar -f "PDF/file" -o "XML/file/to/store/output" -b "PDF/to/store/visualization" --processors SpliceGraphicsItemsAndCharsWithIOU IdentifyMathFromSplicedStructure
  - This makes use of two XML data post-processing steps named at the end of the command
 ### Summary of command-line options
- -f: input PDF file (REQUIRED)

- -o: write output XML to given file name. If missing, XML written to standard output.

- -b: write PDF visualization of bounded boxes to given file name. If missing, no PDF visualization is produced.

- --processors run one or more post-processors listed (currently, only the two used in the second example above are defined).


## Case Study
- **extract_text** script includes a pipeline to extract all the text generated by Symbol Scraper in XML format. We extract all the text line-by-line for each page in the XML file produced by Symbol Scraper.
- To use **extract_text**, run the command below:
  - python3 test.py --xml path/to/xml/file --outputDirectory directory/to/store/output/text --textNme name/of/.txt/file

- For more information, visit "https://gitlab.com/dprl/symbolscraper-server/-/tree/main/"
