# Redis Cloud Setup Guide for SkillShift AI

## 🌐 Step 1: Get Your Redis Cloud URL

From your Redis Cloud dashboard, copy the connection URL. It should look like:

```
redis://default:your_password@redis-12345.c1.ap-southeast-1.redis.redislabs.com:12345
```

## 🔧 Step 2: Set Environment Variable

Choose one of these methods:

### Method A: Create .env file (Recommended)

```bash
# Create .env file in skillshift-ai directory
cd skillshift-ai
echo "REDIS_URL=your_redis_cloud_url_here" > .env
```

### Method B: Export in terminal

```bash
export REDIS_URL="redis://default:your_password@redis-12345.c1.ap-southeast-1.redis.redislabs.com:12345"
```

### Method C: Use the template

```bash
# Copy and edit the config template
cp config.template my_config.sh
# Edit my_config.sh with your Redis Cloud URL
# Then source it:
source my_config.sh
```

## 🚀 Step 3: Start Services

The updated code now automatically detects Redis Cloud vs local Redis:

```bash
# Start all services (will use Redis Cloud if REDIS_URL is set)
./start_all_services.sh
```

You should see:

```
🌐 Connecting to Redis Cloud: redis-12345.c1.ap-southeast-1.redis.redislabs.com:12345
✅ Redis connection established successfully
🌐 Worker connecting to Redis Cloud: redis-12345.c1.ap-southeast-1.redis.redislabs.com:12345
✅ Worker Redis connection established successfully
```

## 🧪 Step 4: Test the Integration

```bash
# Test the async pipeline with Redis Cloud
python test_redis_integration.py
```

## ✅ Verification

1. **Backend health check:**

   ```bash
   curl http://localhost:8000/api/health-isolated
   ```

2. **Create a test job:**

   ```bash
   curl -F file=@screenshot.png http://localhost:8000/api/jobs
   ```

3. **Check job status:**
   ```bash
   curl http://localhost:8000/api/jobs/YOUR_JOB_ID
   ```

## 🔍 Troubleshooting

### Connection Issues

- Verify your Redis Cloud URL is correct
- Check if your Redis Cloud instance is active
- Ensure firewall allows connections to Redis Cloud

### Environment Variable Not Set

```bash
# Check if REDIS_URL is set
echo $REDIS_URL
```

### Worker Connection Issues

- Make sure both backend and worker use the same REDIS_URL
- Restart worker after changing environment variables

## 🌟 Benefits of Redis Cloud

- ✅ **High Availability**: No single point of failure
- ✅ **Persistence**: Jobs survive local machine restarts
- ✅ **Scalability**: Handle more concurrent analysis requests
- ✅ **Monitoring**: Built-in Redis Cloud monitoring dashboard
- ✅ **Security**: SSL/TLS encryption and authentication
