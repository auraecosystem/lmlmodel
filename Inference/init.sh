curl  -i -X POST "https://example.datarobot.com/predApi/v1.0/deployments/<deploymentId>/predictions"  --data-binary $'a,b,c\n1,2,3\n7,8,9\n'
    -H "content-type: text/plain" \
    -H "Authorization: Bearer <API key>" \
