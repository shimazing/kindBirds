package ab.demo;

import java.awt.Color;
import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import Jama.Matrix;
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
import ab.vision.VisionUtils;

public class KindAgent implements Runnable{
	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
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
		currentLevel = level;
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
				state = solve();	
			}
			if (state == GameState.WON) {
				System.out.println("WIN!");
				
				// TODO : sleep??
				
				int score = StateUtil.getScore(ActionRobot.proxy);
				int reward = score - prevScore;
				
				// send gamestate and reward
				try {
					JSONObject stateJson = new JSONObject();
					//Map<String, int[]> stateJson = new HashMap<String, int[]>();
					OutputStream outputStream = socket.getOutputStream();
				
					stateJson.put("gamestate", String.valueOf(1));
					stateJson.put("reward", String.valueOf(reward));
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
				aRobot.loadLevel(currentLevel);;
				firstShot = true;
			}
			else if (state == GameState.LOST) {
				System.out.println("LOSE...");
				
				int score = StateUtil.getScore(ActionRobot.proxy);
				int reward = score - prevScore;
				
				// send gamestate and reward 
				try {
					JSONObject stateJson = new JSONObject();
					//Map<String, int[]> stateJson = new HashMap<String, int[]>();
					OutputStream outputStream = socket.getOutputStream();
					stateJson.put("gamestate", String.valueOf(2));
					stateJson.put("reward", String.valueOf(reward));
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
				aRobot.restartLevel();
				firstShot = true;
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
	
	public GameState solve()
	{
		// zoom out first
		ActionRobot.fullyZoomOut();
		//clickOnce();
		
		// capture image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		
		// process image
		Vision vision = new Vision(screenshot);
		int numBirds = vision.findBirdsRealShape().size();
		int reward;
		ABType birdType = aRobot.getBirdTypeOnSling();
		clickOnce(); // To focus blocks
		if (firstShot) {
			int score =GameStateExtractor.getScoreInGame(screenshot);
			System.out.println("current score : " + String.valueOf(score));
			reward = 0;
		}
		else {
			int score =GameStateExtractor.getScoreInGame(screenshot);
			System.out.println("current score : " + String.valueOf(score));
			reward = score - prevScore;
			prevScore = score;
			System.out.println("reward : " + String.valueOf(reward));
		}
		
		
		// find the slingshot
		Rectangle sling = vision.findSlingshotMBR();
		
		// current game state
		GameState state = aRobot.getState();
		
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
			
			int destWidth = 105;
			int destHeight = 60;
			BufferedImage destImg = scaleDown(screenshot);
			
			int gamestate = getGameState(state);
			try {
				int[] imageRGB = destImg.getRGB(0, 0, destWidth, destHeight, null, 0, destWidth);
	
				JSONObject stateJson = new JSONObject();
				//Map<String, int[]> StateJson =  new HashMap<String, int[]>();
				//JSONObject<String, Object> json = new JSONObject<String, Object>();
				// TODO : send screenshot, gamestate, number of birds, reward
				OutputStream outputStream = socket.getOutputStream();
				//ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
				//ImageIO.write(destImg, "jpg", byteArrayOutputStream);
				
				//stateJson.put("screenshot", Arrays.toString(imageRGB));
				saveScreenshot(destImg, "screenshot.png");
				stateJson.put("gamestate", String.valueOf(gamestate));
				stateJson.put("reward", String.valueOf(reward));
				stateJson.put("birds", String.valueOf(numBirds));
				stateJson.put("birdtype", String.valueOf(getBirdType(birdType)));
				
				System.out.println(stateJson);
				
				OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
				PrintWriter printWriter = new PrintWriter(outputStreamWriter);
				printWriter.println(stateJson);
				printWriter.flush();
				
				BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
				String action = in.readLine();
				
				//printWriter.close();
				
				System.out.println("action");
				System.out.println(action);
				JSONParser parser = new JSONParser();
				JSONObject actionJson = (JSONObject) parser.parse(action);
				float angle = Float.parseFloat((String) actionJson.get("angle"));
				//float power = Float.parseFloat((String) actionJson.get("power"));
				int taptime = Integer.parseInt((String)actionJson.get("taptime"));
				System.out.println("angle : " + String.valueOf(angle));
				System.out.println("taptime : " + String.valueOf(taptime));
				
				Point ref = tp.getReferencePoint(sling);
				Point release = tp.findReleasePoint(sling, angle);
				
				int dx = (int)release.getX() - ref.x;
				int dy = (int)release.getY() - ref.y;
				Shot shot = new Shot(ref.x, ref.y, dx, dy, 0, taptime);
				
				ActionRobot.fullyZoomOut();
				screenshot = ActionRobot.doScreenShot();
				vision = new Vision(screenshot);
				Thread.sleep(1000);
				Rectangle _sling = vision.findSlingshotMBR();
				if(_sling != null)
				{
					double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
					if(scale_diff < 25)
					{
						if(dx < 0)
						{
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
			System.out.println("saving image..");
			ImageIO.write(screenshot,  "png", file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
