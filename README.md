# memnet
Simplest end to end memory network implemented in tensorflow


## File Structure

### dataprovider.py
#### Data Source:
BABI, a simulated reading comprehension dataset with 20 tasks

#### Data Download:
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -zxf tasks_1-20_v1-2.tar.gz

#### Data Structure
We have two classes:
1. GenData:
	Load data for specific task(among the 20 tasks), build dictionary and train/test set, each set is an instance of Data class.

2. Data:
	Data class includes 3 attributes: h(history), q(query) and a(answer). This class implements a function to generate next batch each time.

### memN2N.py
The implementation of End-to-end-memory-network.

### main.py
Call memN2N.py for train.

