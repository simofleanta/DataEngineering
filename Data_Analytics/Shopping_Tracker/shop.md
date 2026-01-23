
* This app will use linear regression to identify spending trends (increasing/decreasing/stable) and classification rules based on purchase frequency and average price to determine the shopper profile (Frequent Buyer, Careful Spender, etc.).
* The app solves the problem that many people are not aware of the daily budget they are using and are not aware of how much daily costs affect their overall personal budget 

It has classification rules:

- Frequent Buyer: freq > 2 AND avg < 30 (buys frequently but small amounts)
- Big Spender: avg > 100 AND freq < 1 (buys rarely but large amounts)
- Variety Shopper: std > 0.8 * avg (high variation between amounts)
- Growing Spender: slope > 0.5 (increasing trend from linear regression)
- Regular Shopper: freq > 1 (constant frequency)
- Careful Spender: all the rest (default)


<img width="1171" height="753" alt="image" src="https://github.com/user-attachments/assets/e40695a0-a56a-41bc-9fe6-f38bb0a1de07" />
<img width="1176" height="754" alt="image" src="https://github.com/user-attachments/assets/e04c7188-1244-40e0-8749-68756870408a" />
