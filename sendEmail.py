#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import smtplib
from email.utils import parseaddr
from email.utils import formataddr
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def send_email(mail_content):
	'''smtp服务器	按需填写
		qq邮箱: smtp.qq.com
		126 邮箱: smtp.126.com
		163 邮箱: smtp.163.com
	'''
	host_server = 'smtp.qq.com'  
	sender = '945823487@qq.com'  	# 发件人邮箱
	code = 'rnigoqtnyvbzbcga'  		# 授权码
	user = '945823487@qq.com'  		# 收件人邮箱
	
	# 邮件信息
	mail_title = '标题'  			# 标题
	senderName = "发件人名称"

	msg = MIMEText(mail_content, _subtype='plain', _charset='utf-8')

	# 格式处理 （防止中文内容邮件会显示乱码）
	msg['Accept-Language'] = 'zh-CN'
	msg['Accept-Charset'] = 'ISO-8859-1,utf-8'

	# 首先用MIMEMultipart()来标识这个邮件由多个部分组成
	msgAtt = MIMEMultipart()
	msgAtt.attach(msg)

	msg['Subject'] = mail_title                             # 邮件主题
	msg['From'] = formataddr(pair=(senderName, sender))     # 发件人和发件人名称
	msg['To'] = user

	smtp = smtplib.SMTP(host_server,587)
	smtp.starttls()
	# 登录--发送者账号和口令
	smtp.login(sender, code)
	# 发送邮件
	smtp.sendmail(sender, user, msg.as_string())
	# 退出
	smtp.quit()


if __name__ == '__main__':
	send_email("done")
