// Simple Dino Game in JavaScript

// Game state
const game = {
    score: 0,
    gameOver: false,
    speed: 5,
    obstacles: [],
    gravity: 0.5,
    jumpPower: 10,
    isJumping: false
};

// Game elements
const canvas = document.createElement('canvas');
canvas.width = 800;
canvas.height = 400;
document.body.appendChild(canvas);

const ctx = canvas.getContext('2d');

// Dino character
const dino = {
    x: 100,
    y: 300,
    width: 50,
    height: 80,
    jumping: false,
    gravity: 0.5,
    velocity: 0,
    onGround: true
};

// Obstacles
function createObstacle() {
    if (game.obstacles.length < 5) {
        game.obstacles.push({
            x: canvas.width,
            y: 300,
            width: 50,
            height: 100
        });
    }
}

// Update game logic
function update() {
    if (game.gameOver) return;

    // Update dino physics
    if (dino.onGround && !dino.jumping) {
        dino.velocity = 0;
        dino.onGround = true;
    }
    
    if (dino.jumping) {
        dino.velocity -= dino.jumpPower;
        dino.y += dino.velocity;
        dino.onGround = false;
    }
    
    // Apply gravity
    if (!dino.onGround) {
        dino.velocity += game.gravity;
        dino.y += dino.velocity;
    }
    
    // Check if dino hits ground
    if (dino.y >= canvas.height - dino.height) {
        dino.y = canvas.height - dino.height;
        dino.velocity = 0;
        dino.onGround = true;
    }
    
    // Move obstacles
    game.obstacles.forEach((obstacle, index) => {
        obstacle.x -= game.speed;
        
        // Check collision
        if (dino.x < obstacle.x + obstacle.width &&
            dino.x + dino.width > obstacle.x &&
            dino.y < obstacle.y + obstacle.height &&
            dino.y + dino.height > obstacle.y) {
            game.gameOver = true;
        }
    });
    
    // Remove obstacles that are off screen
    game.obstacles = game.obstacles.filter(obstacle => obstacle.x > -obstacle.width);
    
    // Create new obstacles periodically
    if (Math.random() < 0.02) {
        createObstacle();
    }
    
    // Update score
    game.score += 1;
}

// Draw game elements
function draw() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw ground
    ctx.fillStyle = '#8B4513';
    ctx.fillRect(0, canvas.height - 50, canvas.width, 50);
    
    // Draw dino
    ctx.fillStyle = '#32CD32';
    ctx.fillRect(dino.x, dino.y, dino.width, dino.height);
    
    // Draw obstacles
    game.obstacles.forEach(obstacle => {
        ctx.fillStyle = '#8B0000';
        ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
    });
    
    // Draw score
    ctx.fillStyle = '#000';
    ctx.font = '20px Arial';
    ctx.fillText(`Score: ${game.score}`, 10, 30);
    
    // Draw game over message
    if (game.gameOver) {
        ctx.fillStyle = 'red';
        ctx.font = '40px Arial';
        ctx.fillText('GAME OVER!', canvas.width/2 - 100, canvas.height/2);
    }
}

// Game loop
function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

// Start the game
gameLoop();

// Add keyboard controls (you can add this to make it interactive)
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' || e.code === 'ArrowUp') {
        if (!game.gameOver) {
            dino.jumping = true;
        }
    }
});

// Add event listeners for mobile touch (optional)
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (!game.gameOver) {
        dino.jumping = true;
    }
});

// Optional: Add a start screen
function showStartScreen() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#FFF';
    ctx.font = '30px Arial';
    ctx.fillText('Dino Game', canvas.width/2 - 100, canvas.height/2 - 50);
    ctx.font = '20px Arial';
    ctx.fillText('Press SPACE or tap to jump', canvas.width/2 - 120, canvas.height/2 + 10);
}

// Show start screen initially
showStartScreen();