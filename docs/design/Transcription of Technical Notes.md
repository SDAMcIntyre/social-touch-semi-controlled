# **Image 1: Validation semi-controlled data**

**ST13-02:**

* **block 01:** forearm position is higher than it should (extract reference).  
* **block 02:**  
  * ref arm \+z shift.  
  * frame mismatch: RGBD vs. stickers.  
* **block 03:**  
  * Ref ARM \+z shift.  
  * hand angle make green sticker mixing with forearm.  
  * \=\> need to implement a separation strategy to stay on the experimenter hand.  
* **block 04:** ref arm \+z shift.  
* **block 05:** ref arm \+z shift.  
* **block 06, 07, 08:** (ditto marks indicating ref arm \+z shift).

**Potential global fix (Part 1):**

* *Side Note:* It seems that the green and yellow stickers spheres aren't aligned with their hand model positions.  
* Improve the position matching of the stickers/hand model.  
* Do an averaging of several forearm ref frame to assess a more exact position.  
  * \-\> Kinect has Z noise.  
* Do some smoothing of the stickers position prior the somatosensory evaluation.  
* Issue with the blue sticker's tracking (or at least Kinect limitation).  
* **G.UI.:** Kinect data seems to lag behind by 1 frame.

# **Image 2: Data Processing & Global Issues**

**Processing Observations:**

* surprisingly, it doesn't seem to change depth and area much...  
* OK calculation must be correct.  
* The problem is when writing the .csv file, it adds "..." between the points, which reduces the list to 6 values.  
  * \[\[x,y,z\], \[x,y,z\], \[x,y,z\], ... \[x,y,z\], \[x,y,z\], \[x,y,z\]\]  
  * **\[That should be fixed\!\]** (from the CSV file investigation).  
  * Also in Kinect unified csv.  
  * Also in Kinect contact & Kinematics data csv.  
  * Generated in compute\_somatosensory\_characteristics.py.

**Blocks Continued:**

* **block 08:** ref arm \+z shift. RGB-D lagg.  
* **block 09:**  
  * ref arm \+z shift (ISSUE A).  
  * RGB-D lagg (B).  
  * blue sticker issue (C).  
* **block 10:** A, B.  
* **block 11:**  
  * A (shift \+z).  
  * B (lagg).  
  * C (stickers) also green sticker has the same behavior.  
* **block 12:** A (+ B?).

**GLOBAL PROBLEM FOR 13-03:**

* green and yellow stickers location on the handmesh make it too big (the 4 digits of Isabella fits in 3 handmesh digits \+ thumb).

# **Image 3: ST13-03 Observations**

**ST13-03:** same issue of forearm \+z shift. (no BO 3, 4\)

* **block 01:**  
  * \[\!\] the RGB-D blue sticker position is on the background (tracking Kinect failed accurately giving the z value).  
  * \-\> only one part of the index has an ok depth value.  
* **block 02:** only \+z shift (reference arm).  
* **block 05:** (ditto) \+ in between touch issue with blue sticker.  
* **block 06:** (ditto) \+ verify synchro.  
* **block 07:**  
  * ref arm \+z shift.  
  * I saw some position of blue stickers which can lack of accuracy. Not sure though.  
  * Kinect RGB-D lagging behind.

**Contact Point Analysis:**

* \[\!\] contact points showed aren't accurate between frames 496 \- 497 (496: good | 497: bad).  
* We must investigate.  
* **Big clue:** if the finger model is not fully covered by the immersed in the forearm.  
  * *(Sketches comparing Frame 496 and Frame 497 overlap)*  
* Hypothesis which seems to not hold.  
* 1132 & 1133 are both like 497, but:  
  * 1132 contains only 6 points (bad).  
  * 1133 contains \+20 (good).