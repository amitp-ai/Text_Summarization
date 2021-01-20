# Data Collection for Text Summarization (Capstone Project)
I explored the following different datasets for text summarization.

## 1. Data Collection
* **CNN/Daily Mail**
    * For Pytorch, can get the dataset easily from <https://huggingface.co/datasets/cnn_dailymail>
    * For Tensorflow, can also easily get the dataset from <https://www.tensorflow.org/datasets/catalog/cnn_dailymail>
    * But for the sake of experience, I decided to collect the data myself from the source that both Huggingface and Tensorflow uses (as cited in their documentation). **In particular, I collected my data from the original place (b) as shown below**
        1. https://github.com/abisee/cnn-dailymail (This is the best place to start/use as it has links/instructions to get processed and unprocessed data)
        2. https://cs.nyu.edu/~kcho/DMQA/ (original unprocessed dataset. Stories has the data for text summarization. Queestions is used for QA task)
           https://github.com/deepmind/rc-data (the script used to generate the data)
        3. https://github.com/becxer/cnn-dailymail/ (suggested by (a) for unprocessed data and how to process it -- it gets it's data from (b))
        4. https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail (suggested by (a) for processed data. Processed data in binary format for Tensorflow. It also has unprocessed data (probably copied from (b))
        5. This actually has the CNN/DM dataset grouped by train/val/test sets <https://github.com/harvardnlp/sent-summary>. Maybe use this instead as it requires less processing work than (a/b)

* **Gigaword dataset**  
  1. Collected data from <https://github.com/harvardnlp/sent-summary> (Tensorflow data (ii) links here too)
  2. For tensorflow, can also get this dataset from <https://www.tensorflow.org/datasets/catalog/gigaword>

* **Opinions dataset** (this is too small though, under 20MB. So not collected/used.)

* **Reddit dataset**
  1. https://www.tensorflow.org/datasets/catalog/reddit (this is very large ~18GB! So not collected/used.)


## 2. Dataset Summary
1. The CNN dataset has 92579 examples and is 392.4MB in size. It tends to have long articles.  
   The CNN dataset md5 checksum (for *.tgz) is: 85ac23a1926a831e8f46a6b8eaf57263

2. The Daily Mail dataset has 219506 examples and is 979.2MB in size. It tends to have moderately long articles.  
   The Daily Mail dataset md5 checksum (for *.tgz) is: f9c5f565e8abe86c38bfa4ae8f96fd72

3. The Gigaword dataset has 3993608 examples and is 939.4MB in size. It tends to have short articles.  
   The Gigaword dataset md5 checksum (for *.tar.gz) is: 064e658dc0d42e0f721d03287a7a9913
   
## 3. Examples

#### **CNN Dataset**
**Article**
The Supreme Court displayed some discomfort on Wednesday as it tried to resolve a testy international custody fight. Deep divisions were apparent from the bench, as the justices tried to sort out competing allegations raised by the parents and proper legal remedies. At issue is whether an appeal of an American judge's order allowing a child to go back to Scotland with the mother becomes moot under the terms of an international treaty since the child is no longer in the United States. Lynne Chafin and her lawyers say the purpose of the Hague Convention treaty was to stop children like her 5-year-old daughter, Eris, from bouncing back and forth between countries over custody. But the girl's father, Jeffrey Chafin, says the custody fight began in American courts where the girl was living at the time, and that his legal claims were never fully heard in those federal courts. A majority on the high court appeared to offer some level of support to the father's legal position, during a tense hour of oral arguments. The word "unfortunate" was mentioned several times by the justices over the entire "he-said, she-said" parental disagreements. "The incentives if you prevail are for the custodial, or the parent with control over the child, to leave immediately," Chief Justice John Roberts told the mother's lawyer. "Even after a motion for a stay has been filed. Get on the first plane out and then you're home free. That seems to me to be a very unfortunate result." Justice Ruth Bader Ginsburg said to the father's lawyer that the purpose of the international treaty was to eliminate "shuttling of children" across courts in different nations, a process she said could take years. An eventual ruling in the case of Chafin v. Chafin could establish an important precedent on the discretion of U.S. courts to decide where children caught in custody fights should stay. In the middle of this case is Eris, who now resides in a remote part of the United Kingdom with her mother. Army Sgt. Jeff Chafin is based at Ft. Stewart, Georgia. He claims she is an unfit mother. She was once arrested for disorderly conduct after a night of drinking and was deported for overstaying her visa. She claims he is manipulative. Back in her home country without her daughter, Lynne filed a "Petition for Return of Child to Scotland" under the 1980 Hague Convention's Civil Aspects of International Child Abduction. It grants parents in general a "right of custody" and a separate "right of access," ensuring the laws of one country are respected in the others. More than 80 nations are a party to the treaty, including the United States and the United Kingdom. Lynne did not see her daughter for 10 months while the custody case played in a federal court in Alabama. Just hours after a federal judge allowed Lynne to take the child to Scotland, the two were on an overseas flight. A federal appeals court later said the matter was out of its hands since the child was now in another country, ruling the issue was moot. Both sides disagree on whether the father moved quickly enough to appeal the immediate removal order in an appropriate court. Now the justices will decide the jurisdictional questions. Michael Manely, the father's attorney, told the justices that the issue was not moot and that the child should be brought back to the United States while courts here resolve the matter. "When you say bring the child back, there's an impediment to that because now this court in Scotland has told her don't-- the child stays here," said Ginsburg. "There's an order that the child not be removed from Scotland." Tougher questions were aimed at Stephen Cullen, attorney for Lynne Chafin, who said "once the doors close on that plane and that child arrives back in Scotland, unless the plane turns round and comes back again, it is all over" on any appeals. "You're not going to suggest all those countries that permit appeals explicitly and stop removals until appeals are finished, that those contracting treaty parties are breaching the convention, are you?" said Justice Sonia Sotomayor skeptically. Roberts -- the only justice with young children -- suggested the father was not given a proper time or opportunity to appeal once the initial removal order was signed by the federal judge in Alabama. "It seems to me," he said, "the best thing is to hold things up briefly, so that the child doesn't go overseas and then have to be brought back, particularly if you have situations where there can be an expeditious appeal. And I think most appellate courts would appreciate the benefit of that." Justice Stephen Breyer said just because the child is in Scotland, does not mean federal courts in the United States still have no role to play. "I don't understand why you want a treaty where the best interests of the child are what's at issue, and then you interpret it in a way that the court of one nation with the child where you have parents from both nations pays no attention at all to what courts in other nations are saying." Jeff Chafin was in the courtroom for the morning arguments, dressed in his Army dress uniform. "I told my little girl I would do everything I could to get her," Sgt. Chafin told CNN National Correspondent Joe Johns this week, "to make sure she's safe." He last visited the girl overseas a few weeks ago. "I don't believe that (the current legal fight) is in the best interest of the child as it's going to go on for years and years to come," Lynne Hales Chafin told CNN. A ruling is due in a few months. The case is Chafin v. Chafin (11-1347).

**Summary**
Case involves girl living with mother in Scotland; husband is in U.S. Custody case tests bounds of international treaty over custody rights. An eventual ruling could establish an important precedent.

---
#### **Daily Mail Dataset**
**Article**
An Oregon couple were lucky to escape injury after their dog triggered a trip wire that was attached to a homemade gun in the woods last week. Mike and Jennifer Colbach. of Portland, were hiking in Forest Park last Thursday when they noticed two men who 'were acting really hinky'. The pair appeared out of nowhere and Mike said they looked like 'the back end of a robbery', according to The Oregonian. While walking into the main trail, Mike and Jennifer noticed a parachute cord on the ground, which they avoided, but their dog Rhoda stepped on. 

Scroll down for video.
Investigating: Mike Colbach (right) and his dog Rhoda returned to a trail near their home outside Forest Park   in Portland, Oregon to show police a trip wire attached to an improvised gun on the trail last week. Lucky escape: Fortunately no one was injured when the dog tripped the wire. The couple realized that something was moving in the trees and, somewhat alarmed, quickly left. However Mike Colbach was troubled about what happened on their regular walking route and went back to investigate a few days later. He discovered the cord was rigged up through the trees to a makeshift gun. The weapon was made from a pipe, that was opened one end and closed the other. At the closed end was a firing pin, and inside the pipe was a shotgun shell. Tripping the parachute cord was meant to trigger a beer bottle that would swing down and hit the firing pin, causing the bullet to fire. However it must have malfunctioned with the Colbach's dog stepped on in it. 'Members of the bomb squad told me, ''Dude, you had a shotgun shell pointed right at your head - twice'','Mr Colbach told The Oregonian after reporting the matter to police. Mike Colbach said the parachute cord was rigged to a three-quarter-inch-diameter pipe - open at one end, closed at the other - attached to a tree. Scene: The incident occurred in Forest Park, a popular hiking and walking spot in Portland, Oregon. Police have taken the weapon as evidence. Authorities said it was rare to see such a device in a public park, as they tend to be more common in rural areas where people are trying to protect something. No other such booby traps have been found in the park. Mr Colbach believes it was the work of the two men they saw.

**Summary**
Device was found at Forest Park in Portland, Oregon. It was a metal pipe with a firing pin loaded with a shotgun shell. Rigged up to a tree so that a wire would trip and a bottle would hit the firing pin. The device malfunction when it was tripped by Mike Colbach's dog, Rhoda. Police have now seized it and are looking for two men seen hanging around the park.

---
#### **Gigaword Dataset**
**Article**
south korea on monday announced sweeping tax reforms , including income and corporate tax cuts to boost growth by stimulating sluggish private consumption and business investment .

**Summary**
skorea announces tax cuts to stimulate economy

---
