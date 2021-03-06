package ab.demo;

import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import ab.demo.other.ActionRobot;
import ab.demo.other.Shot;
import ab.planner.TrajectoryPlanner;
import ab.utils.StateUtil;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.ShowSeg;
import ab.vision.Vision;

public class KindAgent implements Runnable{
	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
	public int minLevel = 1;
	public int maxLevel = 21;
	public boolean training = true;
	public boolean test = false;
	private Map<Integer, Integer> scores = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private boolean firstShot;
	Socket socket;
	int prevScore = 0;
	
	public KindAgent() {
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		firstShot = true;
		randomGenerator = new Random();
		training = false;
		try {
			socket = new Socket("127.0.0.1", 9090);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();
	}
	
	public KindAgent(int level, boolean testing) {
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		firstShot = true;
		randomGenerator = new Random();
		currentLevel = level;
		training = false;
		test = testing;
		try {
			socket = new Socket("127.0.0.1", 9090);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();
	}
	
	public KindAgent(int level) {
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		firstShot = true;
		randomGenerator = new Random();
		maxLevel = level;
		currentLevel = getRandomLevel();
		try {
			socket = new Socket("127.0.0.1", 9090);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();
	}
	
	public KindAgent(int minlevel, int maxlevel) {
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		firstShot = true;
		randomGenerator = new Random();
		minLevel = minlevel;
		maxLevel = maxlevel;
		currentLevel = getRandomLevel();
		try {
			socket = new Socket("127.0.0.1", 9090);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();
		
	}
	
	public void run() {
		aRobot.loadLevel(currentLevel);
		
		while (true) {
			GameState state = aRobot.getState();
			if (state == GameState.PLAYING) {
				try {
					state = solve();
				} catch (NullPointerException e) {
					
					System.out.println("NullPointerException occurred within solve. Try again...");
					continue;
				}
			}
			if (state == GameState.WON) {
				System.out.println("WIN!");
				
				// TODO : sleep??
				
				int score = StateUtil.getScore(ActionRobot.proxy);
				int reward = score - prevScore;
				
				if (!training) {
					if (!scores.containsKey(currentLevel))
						scores.put(currentLevel, score);
					else {
						System.out.println("Old highest score?");
						if (scores.get(currentLevel) < score)
							scores.put(currentLevel, score);
					}
					int totalScore = 0;
					for (Integer key : scores.keySet()) {
						totalScore += scores.get(key);
						System.out.println(" Level " + key + " Score: " + scores.get(key));
					}
					System.out.println("Total Score: " + totalScore);
				}
				
				// send gamestate and reward
				try {
					JSONObject stateJson = createJSON(state, reward, training);
					//Map<String, int[]> stateJson = new HashMap<String, int[]>();
					OutputStream outputStream = socket.getOutputStream();
					/*
					stateJson.put("gamestate", String.valueOf(1));
					stateJson.put("reward", String.valueOf(reward));
					*/
					System.out.println(stateJson);
					
					OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
					PrintWriter printWriter = new PrintWriter(outputStreamWriter);
					printWriter.println(stateJson);
					printWriter.flush();
					
					//printWriter.close();

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				prevScore = 0;
				currentLevel = training? getRandomLevel() : currentLevel + 1;
				if (!test) {
					aRobot.loadLevel(currentLevel);
					tp = new TrajectoryPlanner();
					firstShot = true;
				} else {
					break;
				}
			}
			else if (state == GameState.LOST) {
				System.out.println("LOSE...");
				
				int score = StateUtil.getScore(ActionRobot.proxy);
				int reward = score; // - prevScore;
				
				// send gamestate and reward 
				try {
					JSONObject stateJson = createJSON(state, reward, training);
					//Map<String, int[]> stateJson = new HashMap<String, int[]>();
					OutputStream outputStream = socket.getOutputStream();
					/*
					stateJson.put("gamestate", String.valueOf(2));
					stateJson.put("reward", String.valueOf(reward));
					*/
					System.out.println(stateJson);
					
					OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
					PrintWriter printWriter = new PrintWriter(outputStreamWriter);
					printWriter.println(stateJson);
					printWriter.flush();
					
					//printWriter.close();

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				prevScore = 0;
				if (training) {
					aRobot.restartLevel();
					firstShot = true;
				} else {
					break;
				}
			}
			else if (state == GameState.LEVEL_SELECTION) {
				System.out
				.println("Unexpected level selection page, go to the last current level : "
						+ currentLevel);
				prevScore = 0;
				aRobot.loadLevel(currentLevel);
				firstShot = true;
			} 
			else if (state == GameState.MAIN_MENU) {
				System.out
				.println("Unexpected main menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				prevScore = 0;
				aRobot.loadLevel(currentLevel);
				firstShot = true;
			} 
			else if (state == GameState.EPISODE_MENU) {
				System.out
				.println("Unexpected episode menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				prevScore = 0;
				aRobot.loadLevel(currentLevel);
				firstShot = true;
			}
		}
		/*
		for (int i=0;i<21;i++) {
			aRobot.loadLevel(currentLevel);
			
			//	while (true) {
				GameState state = solve(i);
				ActionRobot.GoFromMainMenuToLevelSelection();
				currentLevel++;
			//}
		}
		*/
	}
	
	public GameState solve() throws NullPointerException
	{
		// zoom out first
		ActionRobot.fullyZoomOut();
		
		// capture image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		
		// process image
		Vision vision = new Vision(screenshot);
		if (aRobot.getState() != GameState.PLAYING)
			return aRobot.getState();
		
		int numBirds = vision.findBirdsRealShape().size();
		int reward;
		ABType birdType = aRobot.getBirdTypeOnSling();
		clickOnce(); // To focus blocks
		int score = GameStateExtractor.getScoreInGame(screenshot);
		if (firstShot) {
			System.out.println("Current score : " + String.valueOf(score));
			reward = 0;
		}
		else {
			System.out.println("Current score : " + String.valueOf(score));
			reward = score - prevScore;
			System.out.println("Reward : " + String.valueOf(reward));
		}
		
		
		// find the slingshot
		Rectangle sling = vision.findSlingshotMBR();
		
		// current game state
		GameState state = aRobot.getState();
		if (state != GameState.PLAYING)
			return state;
		
		
		while (sling == null && state == GameState.PLAYING) {
			System.out.println("No slingshot detected. Please remove pop up or zoom out.");
			ActionRobot.fullyZoomOut();
			//clickOnce();
			screenshot = ActionRobot.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotMBR();
			state = aRobot.getState();
		}
		
		// get all the pigs
		List<ABObject> pigs = vision.findPigsMBR();
		//List<ABObject> blocks = vision.findBlocksMBR();
		
		if (!pigs.isEmpty()) {
			ActionRobot.fullyZoomIn();
			screenshot = ActionRobot.doScreenShot();
			// gray scaled screenshot with real shapes
			ShowSeg.drawRealshape(screenshot);
			
			ActionRobot.fullyZoomOut();
			
			BufferedImage destImg = scaleDown(screenshot);
			
			try {
				JSONObject stateJson = createJSON(state, reward, numBirds, birdType, training);
				//Map<String, int[]> StateJson =  new HashMap<String, int[]>();
				//JSONObject<String, Object> json = new JSONObject<String, Object>();
				OutputStream outputStream = socket.getOutputStream();
				if (training) {
					saveScreenshot(destImg, "screenshot.png");
				}
				else {
					System.out.println("Save screenshow_");
					saveScreenshot(destImg, "screenshot_.png");
				}
				/*
				stateJson.put("gamestate", String.valueOf(gamestate));
				stateJson.put("reward", String.valueOf(reward));
				stateJson.put("birds", String.valueOf(numBirds));
				stateJson.put("birdtype", String.valueOf(getBirdType(birdType)));
				*/
				System.out.println("State : " + stateJson.toString());
				
				OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
				PrintWriter printWriter = new PrintWriter(outputStreamWriter);
				printWriter.println(stateJson);
				printWriter.flush();
				
				BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
				String action = in.readLine();
				
				//printWriter.close();
				
				System.out.println("Action : " + action.toString());
				JSONParser parser = new JSONParser();
				JSONObject actionJson = (JSONObject) parser.parse(action);
				float angle = Float.parseFloat((String) actionJson.get("angle"));
				int taptime = Integer.parseInt((String)actionJson.get("taptime"));
				System.out.println("  Angle : " + String.valueOf(angle));
				System.out.println("  Taptime : " + String.valueOf(taptime));
				
				Point ref = tp.getReferencePoint(sling);
				Point release = tp.findReleasePoint(sling, angle);
				
				int dx = (int)release.getX() - ref.x;
				int dy = (int)release.getY() - ref.y;
				Shot shot = new Shot(ref.x, ref.y, dx, dy, 0, taptime);
				
				ActionRobot.fullyZoomOut();
				screenshot = ActionRobot.doScreenShot();
				vision = new Vision(screenshot);
				// Sleep??
				Thread.sleep(1000);
				Rectangle _sling = vision.findSlingshotMBR();
				if(_sling != null)
				{
					double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
					if(scale_diff < 25)
					{
						if(dx < 0)
						{
							//aRobot.cFastshoot(shot);
							prevScore = score;
							aRobot.cshoot(shot);
							state = aRobot.getState();
							if ( state == GameState.PLAYING )
							{
								screenshot = ActionRobot.doScreenShot();
								vision = new Vision(screenshot);
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, release);
								firstShot = false;
							}
						}
					}
					else
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
				}
				else
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");
		    } catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ParseException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		    //saveScreenshot(destImg, "destImg.png");
			//saveScreenshot(screenshot, "screenshotImage.png");
		}
		
		return state;
	}
	
	private JSONObject createJSON(GameState state, int reward, boolean training) {
		JSONObject stateJson = new JSONObject();
		stateJson.put("gamestate", String.valueOf(getGameState(state)));
		if (training)
			stateJson.put("reward", String.valueOf(reward));
		
		return stateJson;	
	}
	
	private JSONObject createJSON(GameState state, int reward, int numBirds, ABType birdType, boolean training) {
		JSONObject stateJson = new JSONObject();
		stateJson.put("gamestate", String.valueOf(getGameState(state)));
		if (training)
			stateJson.put("reward", String.valueOf(reward));
		stateJson.put("birds", String.valueOf(numBirds));
		stateJson.put("birdtype", String.valueOf(getBirdType(birdType)));
		
		return stateJson;	
	}
	
	private int getRandomLevel() {
		int randomLevel = this.minLevel + this.randomGenerator.nextInt(this.maxLevel - this.minLevel + 1);
		
		System.out.println("Get Random Level");
		System.out.println(randomLevel);
		/*
		randomLevel = this.randomGenerator.nextInt(this.maxLevel - 1) + 1;
		System.out.println(randomLevel);
		randomLevel = this.randomGenerator.nextInt(this.maxLevel - 1) + 1;
		System.out.println(randomLevel);
		*/
		return randomLevel;
	}
		
	private BufferedImage scaleDown(BufferedImage srcImg) {
		int destWidth = 105;
		int destHeight = 60;
		Image imgTarget = srcImg.getScaledInstance(destWidth, destHeight, Image.SCALE_SMOOTH);
		int pixels[] = new int[destWidth * destHeight]; 
		PixelGrabber pg = new PixelGrabber(imgTarget, 0, 0, destWidth, destHeight, pixels, 0, destWidth); 
	    try {
	        pg.grabPixels();
	    } catch (InterruptedException e) {
        	e.printStackTrace();
	    }
	    BufferedImage destImg = new BufferedImage(destWidth, destHeight, BufferedImage.TYPE_INT_RGB); 
	    destImg.setRGB(0, 0, destWidth, destHeight, pixels, 0, destWidth); 
		
	    return destImg;
	}
	
	private int getGameState(GameState state) {
		if (state == GameState.PLAYING)
			return 0;
		else if (state == GameState.WON)
			return 1;
		else if (state == GameState.LOST)
			return 2;
		return -1;
	}
	
	private int getBirdType(ABType bird) {
		if (bird == ABType.RedBird)
			return 0;
		else if (bird == ABType.YellowBird)
			return 1;
		else if (bird == ABType.BlueBird)
			return 2;
		return -1;
	}
	
	private void clickOnce() {
		aRobot.click();
		/*
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		*/
	}
	
	private void saveScreenshot(BufferedImage screenshot, String outputfile) {
		File file = new File(outputfile);
		
		try {
			System.out.println("Saving image..");
			ImageIO.write(screenshot,  "png", file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
