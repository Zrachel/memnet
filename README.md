# memnet
Simplest end to end memory network implemented in tensorflow


## File Structure

### dataprovider.py

#### Data Source:<br/>
BABI, a simulated reading comprehension dataset with 20 tasks

#### Data Download:
```bash
mkdir data && cd data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -zxf tasks_1-20_v1-2.tar.gz
```

#### Data Structure:
 We have two classes:<br/>
 
 1. GenData:<br/>
     Load data for specific task(among the 20 tasks), build dictionary and train/test set, each set is an instance of Data class.<br/>
 2. Data:<br/>
     Data class includes 3 attributes: h(history), q(query) and a(answer). This class implements a function to generate next batch each time.<br/>


### memN2N.py
The implementation of End-to-end-memory-network.

### main.py
Call memN2N.py for train.

